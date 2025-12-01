time: 20251201

# Arxiv Computer Vision Papers - 2025-12-01

## Executive Summary

好的，作为一名专注于计算机视觉和机器学习的研究助理，我将为您提供一份关于近期 Arxiv 计算机视觉领域论文的简明执行摘要。

---

**执行摘要：2025年11月28日 Arxiv 计算机视觉论文精选**

**主要主题与趋势：**

本期 Arxiv 论文集聚焦于**多模态理解与生成**，特别是**视频内容的处理和交互**，以及**视觉生成模型的创新**。多模态语言模型在视频推理、游戏世界建模和运动迁移方面的应用是突出亮点。此外，**可解释性**和**大规模数据驱动的模拟学习**也展现出重要进展。

**亮点与创新：**

*   **视频推理与交互：** "Video-R2" 和 "Video-CoM" 均在视频的多模态推理能力上取得了显著进展，前者强调一致性和接地气的推理，后者则提出了交互式视频推理框架。
*   **生成模型调优与应用：** "Visual Generation Tuning" 和 "DEAL-300K" 分别探索了通用视觉生成模型的微调策略和扩散模型在图像编辑区域定位上的创新应用，后者还引入了大规模数据集。
*   **大规模模拟与游戏AI：** "Hunyuan-GameCraft-2" 和 "SimScale" 展示了在复杂交互环境（如游戏世界）中构建世界模型和通过大规模模拟进行学习的强大能力，预示着更智能的AI代理。
*   **运动表示与生成：** "DisMo" 在解耦运动表示方面提供了新思路，为开放世界运动迁移奠定了基础。

**新兴研究方向与技术：**

*   **视频多模态推理的精细化：** 从简单的视频理解向更深层次的、具备因果和交互能力的推理发展。
*   **生成模型的精细化控制与编辑：** 探索更精确、更可控的图像和视频生成与编辑技术。
*   **大规模、真实感模拟环境的构建与利用：** 为训练更鲁棒、泛化能力更强的AI模型提供新的途径。
*   **解耦表示学习在运动等特定领域的深化应用。**
*   **注意力机制在多模态模型可解释性中的作用日益凸显。**

**建议阅读全文的论文：**

考虑到其对多模态理解、视频交互以及生成模型前沿的贡献，以下论文值得深入阅读：

1.  **"Video-R2: Reinforcing Consistent and Grounded Reasoning in Multimodal Language Models"**: 对于理解多模态模型在视频推理中的一致性和接地气能力至关重要。
2.  **"Video-CoM: Interactive Video Reasoning via Chain of Manipulations"**: 提供了视频交互式推理的新范式，对需要与视频内容进行动态交互的应用具有启发意义。
3.  **"DEAL-300K: Diffusion-based Editing Area Localization with a 300K-Scale Dataset and Frequency-Prompted Baseline"**: 在扩散模型编辑和大规模数据集构建方面具有重要价值，对图像编辑领域的研究者尤为重要。
4.  **"SimScale: Learning to Drive via Real-World Simulation at Scale"**: 展示了大规模模拟在现实世界任务（如自动驾驶）中的潜力，对强化学习和机器人领域的研究者具有参考价值。

---

这份摘要旨在帮助您快速把握本期 Arxiv 论文的核心内容和发展趋势。希望它能为您节省宝贵的研究时间。

---

## Table of Contents

1. [Video-R2: Reinforcing Consistent and Grounded Reasoning in Multimodal Language Models](#2511.23478v1)
2. [Video-CoM: Interactive Video Reasoning via Chain of Manipulations](#2511.23477v1)
3. [Visual Generation Tuning](#2511.23469v1)
4. [Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model](#2511.23429v1)
5. [DisMo: Disentangled Motion Representations for Open-World Motion Transfer](#2511.23428v1)
6. [VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction](#2511.23386v1)
7. [DEAL-300K: Diffusion-based Editing Area Localization with a 300K-Scale Dataset and Frequency-Prompted Baseline](#2511.23377v1)
8. [Optimizing Multimodal Language Models through Attention-based Interpretability](#2511.23375v1)
9. [SimScale: Learning to Drive via Real-World Simulation at Scale](#2511.23369v1)
10. [Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](#2511.23334v1)

---

## Papers

<a id='2511.23478v1'></a>
## [Video-R2: Reinforcing Consistent and Grounded Reasoning in Multimodal Language Models](https://arxiv.org/abs/2511.23478v1)

**Authors:** Muhammad Maaz, Hanoona Rasheed, Fahad Shahbaz Khan, Salman Khan

**Published:** 2025-11-28

**Categories:** cs.CV

**Abstract:**

Reasoning over dynamic visual content remains a central challenge for multimodal large language models. Recent thinking models generate explicit reasoning traces for interpretability; however, their reasoning often appears convincing while being logically inconsistent or weakly grounded in visual evidence. We identify and formalize these issues through two diagnostic metrics: Think Answer Consistency (TAC), which measures the alignment between reasoning and answers, and Video Attention Score (VAS), which captures the extent to which reasoning depends on visual versus textual cues. Analysis across 11 video reasoning benchmarks shows that current models rely heavily on linguistic priors rather than visual content. To address this, we propose a reinforcement learning approach that enhances both temporal precision and reasoning consistency. Our approach combines timestamp aware supervised fine tuning with Group Relative Policy Optimization (GRPO) guided by a novel Temporal Alignment Reward (TAR). This dual step post training stage encourages temporally aligned and causally coherent video reasoning. The resulting model, Video R2, achieves consistently higher TAC, VAS, and accuracy across multiple benchmarks, demonstrating that improvements in temporal alignment and reasoning coherence lead to more accurate and trustworthy video understanding. Our code, dataset, and model will be open sourced.

**Analysis:**

作为一名计算机视觉和机器学习领域的专家，我对这篇论文摘要进行了深入分析，并为您提供以下中文解读：

**1. 论文的主要贡献（2-3句话）**

这篇论文“Video-R2”的核心贡献在于，它识别并解决了当前多模态大语言模型在视频推理中存在的“逻辑不一致”和“视觉证据不足”的问题。作者提出了一种新颖的强化学习方法，通过引入时间对齐奖励（TAR）和结合时间感知监督微调与组相对策略优化（GRPO），显著提升了模型在视频推理任务中的准确性、逻辑一致性以及对视觉信息的依赖程度。

**2. 关键创新或方法论**

论文的关键创新在于其提出的强化学习框架，旨在提升视频推理的“时间精度”和“推理一致性”。具体方法论包括：

*   **诊断指标：** 引入了两个量化指标来评估模型问题：
    *   **Think Answer Consistency (TAC):** 衡量推理过程与最终答案之间的一致性。
    *   **Video Attention Score (VAS):** 评估推理过程对视觉线索和文本线索的依赖程度。
*   **问题诊断：** 通过分析发现现有模型过度依赖语言先验，而对视觉内容依赖不足。
*   **强化学习框架：**
    *   **时间感知监督微调 (Timestamp Aware Supervised Fine Tuning):** 确保模型在微调过程中能够理解和利用视频的时间信息。
    *   **组相对策略优化 (Group Relative Policy Optimization - GRPO):** 一种强化学习优化算法，用于指导模型学习。
    *   **时间对齐奖励 (Temporal Alignment Reward - TAR):** 这是一个新颖的奖励函数，专门设计用于鼓励模型生成在时间上对齐且因果连贯的视频推理。

**3. 对该领域的潜在影响**

这篇论文对多模态大语言模型在视频理解领域具有重要的潜在影响：

*   **提升可信度：** 通过解决逻辑不一致和视觉证据不足的问题，Video-R2有望生成更值得信赖的视频推理结果，这对于需要高可靠性的应用至关重要。
*   **推动更深层次的视觉理解：** 促使模型从单纯的文本关联转向更深入的视觉内容分析，从而实现更强的视觉推理能力。
*   **为视频推理设定新的基准：** 提出的诊断指标和改进方法，可能成为未来视频推理模型研究和评估的新标准。
*   **促进可解释性研究：** 尽管摘要中提到“生成显式推理轨迹”，但其推理的“逻辑不一致”是现有模型的痛点。Video-R2的改进有望使生成的推理轨迹更加可靠和有意义。

**4. 可能受益的相关领域或应用**

这项研究的成果可以广泛应用于以下领域：

*   **视频问答 (Video Question Answering - Video QA):** 提升模型回答复杂视频问题的准确性和逻辑性。
*   **视频摘要 (Video Summarization):** 生成更具连贯性和信息量的视频摘要。
*   **视频内容分析与检索 (Video Content Analysis and Retrieval):** 更好地理解视频内容，实现更精准的检索。
*   **自动驾驶 (Autonomous Driving):** 提升自动驾驶系统对动态场景的理解和预测能力。
*   **机器人交互 (Robotics Interaction):** 使机器人能够更好地理解和响应视频指令或场景。
*   **医疗影像分析 (Medical Imaging Analysis):** 在分析动态医疗影像（如CT、MRI）时，提高诊断的准确性和可解释性。
*   **教育科技 (EdTech):** 用于生成更具解释性的教学视频内容。

**5. 从摘要中推断出的局限性**

尽管摘要展示了积极的成果，但仍可以推断出一些潜在的局限性：

*   **计算成本：** 强化学习方法，尤其是涉及策略优化和奖励设计的，通常计算成本较高，训练和部署可能需要大量的计算资源。
*   **泛化能力：** 虽然在11个基准上取得了成功，但模型在未见过的新颖视频类型或推理任务上的泛化能力仍需进一步验证。
*   **奖励函数设计：** TAR奖励函数的具体设计细节并未在摘要中披露，其有效性可能高度依赖于奖励函数的精细调优。
*   **对“逻辑一致性”的定义：** 摘要中提到“逻辑不一致”，但“逻辑”的定义在复杂视频推理中可能存在主观性，其评估的全面性有待深入研究。
*   **对“视觉证据”的依赖程度：** VAS指标衡量了依赖程度，但如何精确量化和区分“强视觉证据”与“弱视觉证据”仍是一个挑战。

总而言之，这篇论文“Video-R2”通过引入创新的强化学习框架和诊断指标，有效地解决了多模态大语言模型在视频推理中的关键痛点，为构建更强大、更可信赖的视频理解系统奠定了基础。其对时间对齐和推理一致性的关注，使其在计算机视觉领域具有重要的研究价值和应用前景。

**Key Findings:**

- To address this, we propose a reinforcement learning approach that enhances both temporal precision and reasoning consistency.
- Our approach combines timestamp aware supervised fine tuning with Group Relative Policy Optimization (GRPO) guided by a novel Temporal Alignment Reward (TAR).

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23478v1)
- [arXiv](https://arxiv.org/abs/2511.23478v1)

---

<a id='2511.23477v1'></a>
## [Video-CoM: Interactive Video Reasoning via Chain of Manipulations](https://arxiv.org/abs/2511.23477v1)

**Authors:** Hanoona Rasheed, Mohammed Zumri, Muhammad Maaz, Ming-Hsuan Yang, Fahad Shahbaz Khan, Salman Khan

**Published:** 2025-11-28

**Categories:** cs.CV

**Abstract:**

Recent multimodal large language models (MLLMs) have advanced video understanding, yet most still "think about videos" ie once a video is encoded, reasoning unfolds entirely in text, treating visual input as a static context. This passive paradigm creates a semantic bottleneck: models cannot rewatch, refocus, or verify evidence, leading to shallow visual reasoning on tasks requiring fine grained spatio temporal understanding. In this work, we introduce Interactive Video Reasoning, a new paradigm that transforms video into an active cognitive workspace, enabling models to "think with videos". Our model, Video CoM, reasons through a Chain of Manipulations (CoM), performing iterative visual actions to gather and refine evidence. To support this behavior, we construct Video CoM Instruct, an 18K instruction tuning dataset curated for multi step manipulation reasoning. Beyond supervised learning, we further optimize the manipulation policy via reinforcement learning with reasoning aware Group Relative Policy Optimization (GRPO). Unlike prior work that relies solely on sparse answer rewards, our method introduces step level reasoning rewards, guiding the model toward grounded and consistent reasoning. Video CoM achieves strong results across nine video reasoning benchmarks, improving average performance by 3.6 percent over recent state of the art models, while training on only 25K SFT and 3K GRPO video samples, significantly fewer than comparable large scale models. Ablation studies demonstrate that reasoning aware rewards improve both accuracy and interpretability. Code: https://github.com/mbzuai-oryx/Video-CoM

**Analysis:**

好的，这是对论文“Video-CoM: Interactive Video Reasoning via Chain of Manipulations”的全面中文摘要：

**论文题目：** Video-CoM: Interactive Video Reasoning via Chain of Manipulations

**作者：** Hanoona Rasheed, Mohammed Zumri, Muhammad Maaz, Ming-Hsuan Yang, Fahad Shahbaz Khan, Salman Khan

**摘要：**

**1. 主要问题/研究挑战：**
当前的多模态大型语言模型（MLLMs）在视频理解方面取得了显著进展，但它们大多采用“思考视频”（think about videos）的范式。这意味着视频一旦被编码，推理过程完全在文本空间中进行，将视觉输入视为静态上下文。这种被动的感知方式造成了语义瓶颈，使得模型无法重新观看、重新聚焦或验证证据，导致在需要精细时空理解的任务上进行浅层视觉推理。尤其是在需要精细时空理解、精确空间推理或多步证据聚合的任务上，模型容易偏离视觉证据，依赖世界知识，从而导致推理不一致且缺乏视觉基础。

**2. 关键创新/方法贡献：**
为了解决上述问题，论文提出了**交互式视频推理（Interactive Video Reasoning）**的新范式，将视频转化为一个**主动的认知工作空间**，使模型能够“用视频思考”（think with videos）。其核心创新包括：

*   **链式操作（Chain of Manipulations, CoM）机制：** Video-CoM 模型通过一系列**原子视觉操作**（包括 `find-segment`、`find-frame` 和 `spatial-zoom`）来主动与视频交互，以收集和精炼视觉证据。这种操作序列构成了可解释的推理轨迹，每个步骤都以局部证据为基础。
*   **Video-CoM-Instruct 数据集：** 构建了一个包含 18K 样本的**指令调优数据集**，专门用于多步操作推理，旨在引导模型进行操作驱动的视频推理。该数据集精心策划，要求模型执行一个或多个视觉操作来收集证据，模拟人类提取局部信息的方式。
*   **推理感知组相对策略优化（Reasoning-Aware Group Relative Policy Optimization, RA-GRPO）：** 引入了一种新的强化学习目标函数，通过**步级推理奖励**来优化操作策略。与仅依赖稀疏答案奖励的传统方法不同，RA-GRPO 评估中间操作（如时间段 IoU、帧召回率、空间 IoU），即使最终答案不正确，也能为正确的中间步骤提供部分信用，从而引导模型进行更具视觉基础和一致性的推理。

**3. 主要结果及其意义：**
Video-CoM 在九个视频推理基准测试中取得了**显著的性能提升**，平均性能比最近的**最先进模型提高了 3.6%**。尤其值得注意的是，Video-CoM 在 **Video-CoM-Bench**（一个专门为操作中心推理设计的基准）上取得了最大的提升（68.7%），这突显了其交互式推理能力。更重要的是，Video-CoM 仅使用了 **25K SFT 和 3K GRPO 视频样本**进行训练，远少于同类大型模型，显示了其**训练效率**。消融研究表明，推理感知奖励显著提高了模型的**准确性和可解释性**。

**4. 论文中提到的局限性：**
*   **视频中的空间定位：** 在视频中准确地进行空间定位仍然是一个挑战，尤其是在需要识别文本或数字等精细细节时。这需要大规模、高质量的标注数据，而这类数据目前相对稀缺。
*   **视频源的局限性：** 构建操作特定的数据集依赖于具有丰富时空变化和局部细节的视频。低场景多样性的视频可能难以生成需要迭代视觉交互的问题。

**5. 潜在的未来研究方向：**
*   进一步提升模型在复杂空间定位任务上的能力。
*   探索更广泛的视频源，以覆盖更多样化的场景和操作。
*   研究如何将 Video-CoM 的交互式推理范式扩展到其他模态或更复杂的任务。
*   进一步优化训练效率和模型的可扩展性。

**总结：**
“Video-CoM: Interactive Video Reasoning via Chain of Manipulations” 论文提出了一种创新的**交互式视频推理范式**，通过**链式操作（CoM）**和**推理感知奖励（RA-GRPO）**，使 MLLMs 能够像人类一样“用视频思考”，主动收集和精炼视觉证据。该方法在多个视频推理任务上取得了优异的性能，并且训练效率高，为视频理解领域开辟了新的研究方向。该工作强调了从被动感知转向主动交互式推理的重要性，为构建更强大、更具解释性的视频理解模型奠定了基础。

**Key Findings:**

- In this work, we introduce Interactive Video Reasoning, a new paradigm that transforms video into an active cognitive workspace, enabling models to "think with videos".
- Unlike prior work that relies solely on sparse answer rewards, our method introduces step level reasoning rewards, guiding the model toward grounded and consistent reasoning.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23477v1)
- [arXiv](https://arxiv.org/abs/2511.23477v1)

---

<a id='2511.23469v1'></a>
## [Visual Generation Tuning](https://arxiv.org/abs/2511.23469v1)

**Authors:** Jiahao Guo, Sinan Du, Jingfeng Yao, Wenyu Liu, Bo Li, Haoxiang Cao, Kun Gai, Chun Yuan, Kai Wu, Xinggang Wang

**Published:** 2025-11-28

**Categories:** cs.CV

**Abstract:**

Large Vision Language Models (VLMs) effectively bridge the modality gap through extensive pretraining, acquiring sophisticated visual representations aligned with language. However, it remains underexplored whether these representations, optimized for multimodal understanding tasks, harbor an inherent potential for visual generation. In this paper, we propose VGT, Visual Generation Tuning, a novel paradigm designed to stimulate the underlying capabilities of visual generation within any vision language models. By performing efficient visual generation tuning on well-pretrained VLMs, we significantly mitigate the alignment costs and accelerate the convergence of autoregressive modeling in the continuous space (20x speedup). Specifically, we dismiss the entangled pixel-level VAEs designed for diffusion transformers and formulate VGT-AE through aligning the semantic encoders from pretrained VLMs with the latent representations of pixel decoders. In image reconstruction tasks, we achieve 26.67 PSNR and 0.50 rFID at a 28x compression ratio, outperforming specialized VAEs; in visual generation tasks, we achieve state-of-the-art outcomes among autoregressive models, 0.77 on GenEval and 78.73 on DPG-Bench. Furthermore, our proposed VGT showcases significant scaling promise and is versatile for endowing any VLMs trained for multimodal understanding with the capabilities of visual generation, which paves the new avenue to explore next-generation unified multimodal foundation models. Models and codes are available at https://github.com/hustvl/VGT.

**Analysis:**

好的，这是对论文“Visual Generation Tuning”的全面摘要：

**论文题目：** Visual Generation Tuning

**作者：** Jiahao Guo, Sinan Du, Jingfeng Yao, Wenyu Liu, Bo Li, Haoxiang Cao, Kun Gai, Chun Yuan, Kai Wu, Xinggang Wang

**摘要：**

**1. 研究问题/核心问题：**

该论文的核心研究问题是：**大型视觉语言模型（VLMs）在多模态理解任务上经过预训练所获得的视觉表征，是否蕴含了生成图像的内在潜力？** 尽管VLMs在连接视觉和语言模态方面取得了巨大成功，但它们在视觉生成方面的能力尚未得到充分探索。现有的视觉生成方法，特别是基于自回归的模型，通常依赖于与自回归建模不完全对齐的像素级VAE（变分自编码器），这导致了训练的低效和不稳定性。

**2. 主要创新点/方法贡献：**

作者提出了**Visual Generation Tuning (VGT)**，一种新颖的范式，旨在激发现有预训练VLMs中潜在的视觉生成能力。其关键创新点包括：

*   **VGT-AE（Visual Generation Tuning-AutoEncoder）：** 这是一个核心组件，通过将预训练VLM的语义编码器与轻量级像素解码器的潜在空间对齐来实现。这不同于传统扩散模型中用于像素级重建的VAE，而是专注于生成任务。VGT-AE采用两阶段训练策略：
    *   **第一阶段（语义保持重建）：** 使用重建损失和语义自蒸馏损失来优化编码器和解码器，确保重建质量的同时保留语义结构。
    *   **第二阶段（潜在空间正则化）：** 冻结编码器，优化解码器，并引入通道归一化和高斯噪声注入，使潜在空间更符合自回归生成所需的标准高斯先验，提高其分布稳定性和生成友好性。
*   **QueryAR（Query-based Autoregressive）：** 针对自回归生成阶段，提出了一种创新的位置查询机制。它通过在输入序列中交错位置查询和潜在表示，允许在训练时保持因果关系，同时在推理时实现部分并行解码，显著提高了生成效率。
*   **高效的微调范式：** VGT通过高效的视觉生成微调，显著降低了对齐成本，并加速了连续空间自回归建模的收敛速度（最高可达20倍加速）。

**3. 主要结果及其意义：**

*   **重建性能：** VGT-AE在图像重建任务上取得了优异的性能，在28倍压缩比下达到了26.67 PSNR和0.50 rFID，优于专门的VAE。
*   **生成性能：** 在视觉生成任务上，VGT实现了**最先进（SOTA）的自回归模型性能**，在GenEval上达到0.77，在DPG-Bench上达到78.73。
*   **数据效率：** VGT在仅使用25M训练样本的情况下，就取得了与大型扩散模型（如SDXL、SD3-Medium）相媲美的性能，这挑战了传统观念，即自回归模型需要海量数据才能达到高质量生成。
*   **通用性与可扩展性：** VGT被证明对各种预训练VLMs（如Qwen2.5-VL和InternVL3）都具有**高度的通用性**，能够赋予它们视觉生成能力，为构建下一代统一的多模态基础模型铺平了道路。
*   **效率提升：** QueryAR通过位置查询机制，在保持生成质量的同时，实现了显著的推理加速（例如，4倍加速下仍能保持竞争力）。

**4. 论文中提到的局限性：**

*   **重建与生成的权衡：** 论文的消融研究（Section 4.4.3）表明，在VGT-AE的训练中，存在重建保真度和生成能力之间的权衡。过度优化重建可能导致生成性能下降，反之亦然。虽然VGT-AE通过两阶段训练实现了良好的平衡，但这种权衡仍然存在。
*   **模型大小与性能：** 虽然VGT展示了良好的可扩展性，但不同大小的模型在不同任务上的表现仍有差异。例如，在Table 6中，0.6B的模型在GenEval和DPG-Bench上的得分低于1.6B的模型。
*   **对齐的复杂性：** 尽管VGT大大降低了对齐成本，但论文也提到，即使在AE和LLM来自不同VLM家族的“不匹配”情况下，VGT-AE仍然优于VAE基线，这表明跨模态对齐仍然是影响性能的一个因素。

**5. 潜在的未来研究方向：**

*   **更精细的生成控制：** 尽管VGT在生成方面表现出色，但未来可以探索更精细的控制机制，以实现对生成图像的风格、内容或特定属性的更精确调控。
*   **多模态融合的进一步探索：** VGT为统一多模态模型提供了新的视角。未来的工作可以进一步探索如何更有效地融合视觉和语言信息，以实现更强大的理解和生成能力。
*   **更广泛的模态扩展：** VGT的范式可以扩展到其他模态，例如音频或视频生成，以构建更全面的多模态生成模型。
*   **更高效的训练和推理技术：** 尽管VGT已经实现了显著的效率提升，但持续探索更高效的训练算法和推理优化技术仍然是重要的研究方向。
*   **模型安全与伦理考量：** 随着生成模型能力的增强，研究其潜在的滥用风险以及开发相应的安全和伦理对策也变得日益重要。

**总结：**

“Visual Generation Tuning”论文提出了一种开创性的方法，成功地将大型视觉语言模型（VLMs）的强大视觉理解能力转化为高效的视觉生成能力。通过引入VGT-AE和QueryAR等创新组件，该方法不仅在重建和生成任务上取得了最先进的性能，而且显著提高了数据效率和训练速度。VGT范式的通用性和可扩展性使其成为构建下一代统一多模态基础模型的有力候选方案，为该领域的研究开辟了新的方向。

**Key Findings:**

- In this paper, we propose VGT, Visual Generation Tuning, a novel paradigm designed to stimulate the underlying capabilities of visual generation within any vision language models.
- In image reconstruction tasks, we achieve 26.67 PSNR and 0.50 rFID at a 28x compression ratio, outperforming specialized VAEs; in visual generation tasks, we achieve state-of-the-art outcomes among autoregressive models, 0.77 on GenEval and 78.73 on DPG-Bench.
- Furthermore, our proposed VGT showcases significant scaling promise and is versatile for endowing any VLMs trained for multimodal understanding with the capabilities of visual generation, which paves the new avenue to explore next-generation unified multimodal foundation models.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23469v1)
- [arXiv](https://arxiv.org/abs/2511.23469v1)

---

<a id='2511.23429v1'></a>
## [Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model](https://arxiv.org/abs/2511.23429v1)

**Authors:** Junshu Tang, Jiacheng Liu, Jiaqi Li, Longhuang Wu, Haoyu Yang, Penghao Zhao, Siruis Gong, Xiang Yuan, Shuai Shao, Qinglin Lu

**Published:** 2025-11-28

**Categories:** cs.CV

**Abstract:**

Recent advances in generative world models have enabled remarkable progress in creating open-ended game environments, evolving from static scene synthesis toward dynamic, interactive simulation. However, current approaches remain limited by rigid action schemas and high annotation costs, restricting their ability to model diverse in-game interactions and player-driven dynamics. To address these challenges, we introduce Hunyuan-GameCraft-2, a new paradigm of instruction-driven interaction for generative game world modeling. Instead of relying on fixed keyboard inputs, our model allows users to control game video contents through natural language prompts, keyboard, or mouse signals, enabling flexible and semantically rich interaction within generated worlds. We formally defined the concept of interactive video data and developed an automated process to transform large-scale, unstructured text-video pairs into causally aligned interactive datasets. Built upon a 14B image-to-video Mixture-of-Experts(MoE) foundation model, our model incorporates a text-driven interaction injection mechanism for fine-grained control over camera motion, character behavior, and environment dynamics. We introduce an interaction-focused benchmark, InterBench, to evaluate interaction performance comprehensively. Extensive experiments demonstrate that our model generates temporally coherent and causally grounded interactive game videos that faithfully respond to diverse and free-form user instructions such as "open the door", "draw a torch", or "trigger an explosion".

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model**

**1. 论文的主要贡献（2-3句话的简洁总结）**

该论文提出了Hunyuan-GameCraft-2，一种新颖的指令驱动交互式游戏世界建模范式。它通过自然语言指令、键盘或鼠标信号实现对生成游戏视频内容的灵活控制，克服了现有方法在动作模式僵化和标注成本高昂方面的限制。该模型能够生成时间连贯且因果关系明确的交互式游戏视频，并能忠实响应多样化的自由形式用户指令。

**2. 关键创新或方法论**

*   **指令驱动的交互范式 (Instruction-Driven Interaction Paradigm):** 这是最核心的创新。不同于以往依赖预设的、僵化的动作模式（如固定键盘输入），Hunyuan-GameCraft-2允许用户使用自然语言（如“开门”、“拔出火把”、“触发爆炸”）来控制游戏世界的动态。这种方式极大地提升了交互的灵活性和语义丰富度。
*   **交互式视频数据的定义与自动化处理:** 论文正式定义了“交互式视频数据”的概念，并开发了一种自动化流程，将大规模、非结构化的文本-视频对转化为因果对齐的交互式数据集。这解决了构建高质量交互式数据集的难题，为模型训练提供了基础。
*   **基于14B MoE基础模型的文本驱动交互注入机制:** 模型构建在一个强大的140亿参数的图像到视频MoE（Mixture-of-Experts）基础模型之上。关键在于其“文本驱动的交互注入机制”，该机制能够对相机运动、角色行为和环境动态进行精细化控制，将自然语言指令有效地转化为视频中的具体动作和变化。
*   **交互式基准测试 (InterBench):** 论文引入了一个专门用于评估交互性能的基准测试集InterBench，这为衡量和比较不同交互式视频生成模型的能力提供了一个标准化的平台。

**3. 对该领域的潜在影响**

*   **推动生成式世界模型的进步:** Hunyuan-GameCraft-2将生成式世界模型从静态场景合成和简单动态模拟提升到了一个全新的水平，使其能够理解并响应复杂的、语义丰富的用户指令，从而创造出更具沉浸感和可玩性的虚拟世界。
*   **降低交互式内容创作门槛:** 通过自然语言指令控制，极大地降低了用户创建和编辑交互式视频内容的门槛，使得非专业人士也能轻松地生成复杂的交互场景。
*   **促进人机交互在多模态领域的融合:** 该研究是多模态学习（文本、视觉、动作）在游戏领域深度融合的典范，为未来更自然、更直观的人机交互方式提供了新的思路。
*   **为游戏开发和虚拟现实/增强现实 (VR/AR) 领域带来新机遇:** 这种技术可以极大地加速游戏关卡设计、NPC行为模拟以及VR/AR场景的动态生成，为游戏开发者和VR/AR内容创作者提供强大的工具。

**4. 可能受益的相关领域或应用**

*   **游戏开发:** 自动化关卡设计、动态NPC行为生成、交互式剧情生成、游戏测试自动化。
*   **虚拟现实/增强现实 (VR/AR):** 动态生成沉浸式VR/AR体验、交互式虚拟导览、虚拟培训模拟。
*   **内容创作:** 自动化视频编辑、交互式故事叙述、教育内容生成。
*   **机器人学:** 学习和执行自然语言指令来与环境交互，尤其是在模拟环境中进行训练。
*   **数字人与虚拟助手:** 创造更具交互性和响应性的虚拟角色。
*   **电影与动画制作:** 快速生成概念场景、模拟复杂动作。

**5. 从摘要中可以推断出的局限性**

*   **计算资源需求:** 基于140亿参数的MoE模型，可以推断其训练和推理需要巨大的计算资源，这可能限制其在资源受限环境下的部署。
*   **对训练数据的依赖:** 尽管论文提到了自动化处理，但模型的性能仍然高度依赖于训练数据的质量和规模，特别是因果对齐的交互式数据集。如果数据存在偏差或不足，模型可能在某些指令上表现不佳。
*   **“理解”的深度:** 虽然模型能响应指令，但其对指令的“理解”程度可能仍是有限的。对于非常抽象、模糊或需要深层推理的指令，模型可能难以准确执行。摘要中给出的例子（“开门”、“拔出火把”、“触发爆炸”）相对具体，更复杂的指令可能面临挑战。
*   **因果关系的鲁棒性:** 尽管论文强调了“因果关系”，但生成视频中的因果关系是否在所有情况下都绝对鲁棒和可信，仍需进一步验证。例如，是否会生成“开门”但门没有打开，或者“触发爆炸”但没有发生爆炸的错误。
*   **泛化能力:** 模型在未见过的新颖游戏环境或指令组合上的泛化能力如何，摘要中并未详细说明，这通常是大型生成模型面临的挑战。
*   **交互的实时性:** 对于需要实时交互的游戏应用，模型的推理速度是否足够快以满足实时性要求，这一点在摘要中没有明确提及。

总而言之，Hunyuan-GameCraft-2在指令驱动的交互式视频生成领域取得了显著进展，其核心创新在于将自然语言指令转化为对游戏世界动态的精细控制，并为此构建了相应的数据集和评估框架。这为计算机视觉和多模态AI领域带来了令人兴奋的可能性，尤其是在游戏、VR/AR和内容创作等应用场景。

**Key Findings:**

- To address these challenges, we introduce Hunyuan-GameCraft-2, a new paradigm of instruction-driven interaction for generative game world modeling.
- We introduce an interaction-focused benchmark, InterBench, to evaluate interaction performance comprehensively.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23429v1)
- [arXiv](https://arxiv.org/abs/2511.23429v1)

---

<a id='2511.23428v1'></a>
## [DisMo: Disentangled Motion Representations for Open-World Motion Transfer](https://arxiv.org/abs/2511.23428v1)

**Authors:** Thomas Ressler-Antal, Frank Fundel, Malek Ben Alaya, Stefan Andreas Baumann, Felix Krause, Ming Gui, Björn Ommer

**Published:** 2025-11-28

**Categories:** cs.CV

**Abstract:**

Recent advances in text-to-video (T2V) and image-to-video (I2V) models, have enabled the creation of visually compelling and dynamic videos from simple textual descriptions or initial frames. However, these models often fail to provide an explicit representation of motion separate from content, limiting their applicability for content creators. To address this gap, we propose DisMo, a novel paradigm for learning abstract motion representations directly from raw video data via an image-space reconstruction objective. Our representation is generic and independent of static information such as appearance, object identity, or pose. This enables open-world motion transfer, allowing motion to be transferred across semantically unrelated entities without requiring object correspondences, even between vastly different categories. Unlike prior methods, which trade off motion fidelity and prompt adherence, are overfitting to source structure or drifting from the described action, our approach disentangles motion semantics from appearance, enabling accurate transfer and faithful conditioning. Furthermore, our motion representation can be combined with any existing video generator via lightweight adapters, allowing us to effortlessly benefit from future advancements in video models. We demonstrate the effectiveness of our method through a diverse set of motion transfer tasks. Finally, we show that the learned representations are well-suited for downstream motion understanding tasks, consistently outperforming state-of-the-art video representation models such as V-JEPA in zero-shot action classification on benchmarks including Something-Something v2 and Jester. Project page: https://compvis.github.io/DisMo

**Analysis:**

好的，作为一名计算机视觉和机器学习领域的专家，我将根据您提供的论文摘要进行深入分析。

**论文摘要分析：DisMo: Disentangled Motion Representations for Open-World Motion Transfer**

**1. 论文的主要贡献 (2-3句话)**

该论文提出了DisMo，一种新颖的范式，旨在从原始视频数据中学习抽象的运动表示。其核心贡献在于能够将运动信息与内容（如外观、身份、姿态）完全解耦，从而实现“开放世界”的运动迁移，即在语义不相关的实体之间进行运动迁移，无需预先建立对应关系，甚至跨越不同类别。DisMo通过解耦运动语义与外观，解决了现有方法在运动保真度、提示遵循度、过拟合和漂移等方面的不足，并能与现有视频生成器无缝集成。

**2. 关键创新或方法论**

*   **图像空间重建目标 (Image-space Reconstruction Objective):** 这是DisMo学习抽象运动表示的核心机制。通过在图像空间进行重建，模型被训练来捕捉视频中的动态变化，而忽略静态的视觉内容。
*   **运动与内容的解耦 (Disentangled Motion from Content):** 这是DisMo最关键的创新点。它明确地将运动信息（如动作的发生、方向、速度）与静态的视觉特征（如物体的外观、身份、姿态）分离开来。这种解耦使得运动表示是通用的，不依赖于特定的物体或场景。
*   **开放世界运动迁移 (Open-World Motion Transfer):** 基于运动与内容的解耦，DisMo能够实现跨越语义界限的运动迁移。这意味着可以将一个物体的运动模式应用到另一个完全不同的物体上，而无需事先知道它们之间的对应关系。例如，可以将一个人的跑步动作迁移到一个非生物物体上。
*   **轻量级适配器集成 (Lightweight Adapters for Integration):** DisMo的运动表示可以通过轻量级适配器与任何现有的视频生成器（如T2V, I2V模型）结合。这使得研究人员和内容创作者能够轻松地利用DisMo的运动迁移能力，并受益于未来视频生成技术的进步。

**3. 对该领域的潜在影响**

DisMo的提出可能对计算机视觉领域的视频生成、内容创作和运动理解产生深远影响：

*   **提升视频生成的可控性与灵活性:** 当前的视频生成模型虽然强大，但在精细控制运动方面仍有局限。DisMo提供的解耦运动表示将极大地增强用户对视频中动作的控制能力，实现更具创造性和个性化的视频内容。
*   **推动跨模态和跨领域的应用:** 开放世界运动迁移的能力将为虚拟现实、增强现实、游戏开发、动画制作等领域带来新的可能性，例如，可以轻松地将现实世界的动作捕捉数据应用到虚拟角色上，或者将一种物体的运动风格迁移到另一种物体上。
*   **加速视频理解研究:** DisMo学习到的通用运动表示，在零样本动作分类等下游任务中表现出色，表明其对运动语义的深刻理解。这有望推动视频理解模型在更广泛、更具挑战性的场景下的性能提升。
*   **降低内容创作门槛:** 通过将复杂的运动迁移任务简化为使用解耦的运动表示，DisMo有望降低内容创作者的技术门槛，使更多人能够轻松地创作高质量的动态视频。

**4. 可能受益的相关领域或应用**

*   **视频生成与编辑:** T2V, I2V模型，以及任何需要精细控制视频中动作的生成任务。
*   **内容创作:** 电影、动画、广告、社交媒体视频的制作。
*   **虚拟现实 (VR) 和增强现实 (AR):** 创建更逼真、更具交互性的虚拟环境和角色动画。
*   **游戏开发:** 角色动画的生成和迁移，实现更丰富的游戏动作。
*   **机器人学:** 运动规划和模仿学习，将人类的运动技能迁移到机器人上。
*   **人机交互:** 设计更自然的交互方式，例如通过手势迁移来控制虚拟助手。
*   **体育分析:** 动作识别、运动员表现分析和训练。
*   **医学影像:** 分析和模拟生物体的运动。

**5. 从摘要中可以推断出的局限性**

尽管摘要中强调了DisMo的优势，但仍可以推断出一些潜在的局限性：

*   **对“原始视频数据”的依赖:** 论文提到“直接从原始视频数据中学习”，这意味着模型的性能可能在很大程度上依赖于训练数据的质量和多样性。如果训练数据中缺乏某些类型的运动或场景，模型在处理这些情况时可能会遇到困难。
*   **“抽象运动表示”的解释性:** 虽然表示是抽象的，但其具体的“语义”和“可解释性”可能仍是一个研究方向。如何精确地理解和控制这些抽象表示，使其符合人类的直观理解，可能需要进一步的研究。
*   **计算成本:** 学习抽象表示通常需要大量的计算资源和时间。虽然摘要提到了“轻量级适配器”，但DisMo本身的训练过程可能仍然是计算密集型的。
*   **“开放世界”的边界:** 尽管论文声称是“开放世界”运动迁移，但“开放世界”的定义和边界是模糊的。在某些极端情况下，例如迁移非常复杂、高度依赖特定物理属性的运动，或者在完全不相关的领域之间迁移，可能仍然会遇到挑战。
*   **对“提示遵循度”的权衡:** 摘要提到“Unlike prior methods, which trade off motion fidelity and prompt adherence”，暗示DisMo在某些情况下可能仍然需要在运动保真度和对特定提示（例如文本描述）的遵循度之间进行权衡，尽管它声称做得更好。

总而言之，DisMo是一项非常有前景的研究，它通过创新的解耦方法，为视频生成和理解领域带来了新的可能性，尤其是在实现通用、灵活的运动迁移方面。其对运动与内容的分离处理，是解决当前视频模型局限性的关键一步。

**Key Findings:**

- To address this gap, we propose DisMo, a novel paradigm for learning abstract motion representations directly from raw video data via an image-space reconstruction objective.
- Unlike prior methods, which trade off motion fidelity and prompt adherence, are overfitting to source structure or drifting from the described action, our approach disentangles motion semantics from appearance, enabling accurate transfer and faithful conditioning.
- We demonstrate the effectiveness of our method through a diverse set of motion transfer tasks.
- Finally, we show that the learned representations are well-suited for downstream motion understanding tasks, consistently outperforming state-of-the-art video representation models such as V-JEPA in zero-shot action classification on benchmarks including Something-Something v2 and Jester.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23428v1)
- [arXiv](https://arxiv.org/abs/2511.23428v1)

---

<a id='2511.23386v1'></a>
## [VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction](https://arxiv.org/abs/2511.23386v1)

**Authors:** Sinan Du, Jiahao Guo, Bo Li, Shuhao Cui, Zhengzhuo Xu, Yifu Luo, Yongxian Wei, Kun Gai, Xinggang Wang, Kai Wu, Chun Yuan

**Published:** 2025-11-28

**Categories:** cs.CV

**Abstract:**

Unifying multimodal understanding, generation and reconstruction representation in a single tokenizer remains a key challenge in building unified models. Previous research predominantly attempts to address this in a dual encoder paradigm, e.g., utilizing the separate encoders for understanding and generation respectively or balancing semantic representations and low-level features with contrastive loss. In this paper, we propose VQRAE, a Vector Quantization version of Representation AutoEncoders, which pioneers the first exploration in unified representation to produce Continuous semantic features for image understanding and Discrete tokens for visual generation within a unified tokenizer. Specifically, we build upon pretrained vision foundation models with a symmetric ViT decoder and adopt a two-stage training strategy: first, it freezes the encoder and learns a high-dimensional semantic VQ codebook with pixel reconstruction objective; then jointly optimizes the encoder with self-distillation constraints. This design enables negligible semantic information for maintaining the ability of multimodal understanding, discrete tokens that are compatible for generation and fine-grained reconstruction. Besides, we identify the intriguing property in quantizing semantic encoders that rely on high-dimensional codebook in contrast to the previous common practice of low-dimensional codebook in image reconstruction. The semantic VQ codebook can achieve a 100% utilization ratio at a dimension of 1536. VQRAE presents competitive performance on several benchmarks of visual understanding, generation and reconstruction with promising scaling property in the autoregressive paradigm for its discrete merits.

**Analysis:**

好的，这是对论文“VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction”的全面中文摘要：

**论文题目：** VQRAE: 用于多模态理解、生成和重建的表示量化自编码器

**作者：** Sinan Du, Jiahao Guo, Bo Li, Shuhao Cui, Zhengzhuo Xu, Yifu Luo, Yongxian Wei, Kun Gai, Xinggang Wang, Kai Wu, Chun Yuan

**摘要：**

**1. 研究问题/核心挑战：**
当前构建统一的多模态模型面临的关键挑战在于如何在一个单一的“tokenizer”（分词器）中统一表示理解、生成和重建任务。现有的方法通常采用双编码器范式，即为理解和生成任务分别设置独立的编码器，或者通过对比损失来平衡语义表示和低级特征。然而，这些方法往往增加了模型复杂度，阻碍了不同表示之间的深层交互，并且需要巨大的批次大小来平衡损失冲突。

**2. 主要创新点/方法贡献：**
本文提出了VQRAE（Vector Quantization Representation AutoEncoders），一种创新的表示量化自编码器，旨在解决上述挑战。其核心贡献在于：
*   **统一的Tokenizer：** VQRAE 是首个能够在一个统一的 tokenizer 中同时产生用于图像理解的**连续语义特征**和用于视觉生成与重建的**离散Tokens**的模型。
*   **两阶段训练策略：**
    *   **第一阶段：** 冻结预训练的视觉基础模型（VFMs）编码器，学习一个高维度的语义 VQ 码本，并使用像素重建目标进行训练。
    *   **第二阶段：** 解冻 VFMs 编码器，并引入自蒸馏损失来增强语义理解能力，同时继续优化以实现精细的重建。
*   **高维语义 VQ 码本：** 论文强调了量化语义编码器时，采用高维度码本（例如 1536 维）的优势，这与以往在图像重建中常用低维度码本的做法形成对比。VQRAE 实现了码本的 100% 利用率。
*   **无卷积结构：** VQRAE 采用对称的 ViT 解码器，避免了卷积像素编码器的使用，简化了模型设计。

**3. 主要结果与意义：**
VQRAE 在多个视觉理解、生成和重建的基准测试中取得了具有竞争力的性能。其主要优势在于：
*   **性能优势：** 在多模态理解任务上，VQRAE 显著优于其他统一 tokenizer 方法，并且在某些方面超越了双编码器方法。在重建任务上，VQRAE 也展现了出色的性能。
*   **效率提升：** VQRAE 简化了模型结构，降低了训练开销，并且无需为 tokenizer 进行额外的预训练或微调，即可直接集成到现有的 MLLMs 中。
*   **可扩展性：** VQRAE 的离散特性使其在自回归范式下具有良好的可扩展性，为构建更强大的统一多模态模型开辟了新途径。
*   **码本利用率：** 成功训练了一个高维度（1536）且利用率高达 100% 的语义 VQ 码本，这在以往的研究中是罕见的。

**4. 论文中提到的局限性：**
*   **理解与重建的权衡：** VQRAE 在平衡理解和重建性能方面仍有改进空间，可能存在一定的折衷。
*   **量化损失：** 离散 tokenizer 的固有量化损失使其在与最先进的连续 VAEs 竞争时面临挑战。
*   **生成质量：** 在生成方面，尤其是在处理手指、人脸等细节时，仍然存在一些伪影，可能需要后处理或更深入的训练来解决。

**5. 未来研究方向：**
*   **更有效的权衡方法：** 探索更有效的方法来平衡理解和重建性能，以最小化对理解能力的影响。
*   **生成与理解的增强：** 研究如何利用生成和重建能力来进一步增强模型的理解能力。
*   **改进生成质量：** 进一步提升生成图像的质量，特别是在空间关系、纹理渲染以及处理人脸和手指伪影方面。
*   **集成与模型扩展：** 探索如何将 VQRAE 产生的表示集成到更广泛的任务中，并研究高效的模型扩展策略。

总而言之，VQRAE 是一项重要的研究成果，它通过提出一种新颖的统一 tokenizer 架构，有效地解决了多模态理解、生成和重建任务之间的权衡问题，并为未来更强大、更高效的统一多模态模型奠定了基础。

**Key Findings:**

- In this paper, we propose VQRAE, a Vector Quantization version of Representation AutoEncoders, which pioneers the first exploration in unified representation to produce Continuous semantic features for image understanding and Discrete tokens for visual generation within a unified tokenizer.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23386v1)
- [arXiv](https://arxiv.org/abs/2511.23386v1)

---

<a id='2511.23377v1'></a>
## [DEAL-300K: Diffusion-based Editing Area Localization with a 300K-Scale Dataset and Frequency-Prompted Baseline](https://arxiv.org/abs/2511.23377v1)

**Authors:** Rui Zhang, Hongxia Wang, Hangqing Liu, Yang Zhou, Qiang Zeng

**Published:** 2025-11-28

**Categories:** cs.CV

**Abstract:**

Diffusion-based image editing has made semantic level image manipulation easy for general users, but it also enables realistic local forgeries that are hard to localize. Existing benchmarks mainly focus on the binary detection of generated images or the localization of manually edited regions and do not reflect the properties of diffusion-based edits, which often blend smoothly into the original content. We present Diffusion-Based Image Editing Area Localization Dataset (DEAL-300K), a large scale dataset for diffusion-based image manipulation localization (DIML) with more than 300,000 annotated images. We build DEAL-300K by using a multi-modal large language model to generate editing instructions, a mask-free diffusion editor to produce manipulated images, and an active-learning change detection pipeline to obtain pixel-level annotations. On top of this dataset, we propose a localization framework that uses a frozen Visual Foundation Model (VFM) together with Multi Frequency Prompt Tuning (MFPT) to capture both semantic and frequency-domain cues of edited regions. Trained on DEAL-300K, our method reaches a pixel-level F1 score of 82.56% on our test split and 80.97% on the external CoCoGlide benchmark, providing strong baselines and a practical foundation for future DIML research.The dataset can be accessed via https://github.com/ymhzyj/DEAL-300K.

**Analysis:**

好的，这是对论文“DEAL-300K: Diffusion-based Editing Area Localization with a 300K-Scale Dataset and Frequency-Prompted Baseline”的全面中文摘要：

**论文题目：** DEAL-300K: 基于扩散的编辑区域定位，包含一个300K规模的数据集和频率提示基线

**作者：** Rui Zhang, Hongxia Wang, Hangqing Liu, Yang Zhou, Qiang Zeng

**摘要：**

**1. 主要问题/研究问题：**
随着扩散模型在图像编辑领域的广泛应用，用户可以轻松进行语义级别的图像编辑，但也带来了更逼真、更难检测的局部伪造图像。现有的基准数据集和方法主要关注二值化的生成图像检测或手动编辑区域的定位，未能充分反映扩散模型编辑的特点，即编辑内容往往能平滑地融入原始图像，缺乏明显的伪影。因此，如何准确地定位扩散模型生成的编辑区域（Diffusion-based Image Manipulation Localization, DIML）是一个亟待解决的问题。

**2. 关键创新/方法贡献：**
*   **DEAL-300K 数据集：** 作者构建了一个大规模的、专门用于 DIML 任务的数据集，包含超过30万张标注图像。该数据集的生成过程具有创新性：
    *   **多模态大语言模型（MLLM）驱动的指令生成：** 利用微调后的 Qwen-VL 模型，根据图像内容自动生成高质量的编辑指令，确保指令的语义一致性。
    *   **无掩码扩散模型进行图像编辑：** 使用 InstructPix2Pix 等无掩码扩散模型生成编辑后的图像，更贴近实际的编辑流程，并避免了对掩码的依赖。
    *   **主动学习与变化检测相结合的像素级标注：** 采用 SAM-CD 模型进行变化检测，并结合主动学习策略，实现了高效且准确的像素级标注，大大减少了人工标注的成本。
*   **多频段提示微调（MFPT）框架：** 作者提出了一种新颖的定位框架，该框架：
    *   **利用冻结的视觉基础模型（VFM）：** 借鉴了大型视觉基础模型强大的先验知识，但通过参数高效的微调（PEFT）方式，避免了全参数微调的计算开销和灾难性遗忘。
    *   **融合语义和频率域信息：** 引入了“频率输入提示器”（FInP）和“特征频率提示器”（FFrP）两个核心组件。FInP 关注图像的低级纹理细节（高频信息），而 FFrP 则通过多头自注意力机制融合高频和低频信息，以捕捉扩散编辑中细微的语义和纹理异常。
    *   **处理扩散编辑的独特性：** 针对扩散模型编辑的特点，MFPT 框架能够捕捉到即使是平滑融入的编辑区域，也可能存在的频率域上的细微变化。

**3. 主要结果及其意义：**
*   **数据集性能：** DEAL-300K 数据集在作者提出的 MFPT 框架下，在内部测试集上达到了 82.56% 的像素级 F1 分数，在外部 CoCoGlide 基准测试集上达到了 80.97% 的像素级 F1 分数，显著优于现有方法，为 DIML 研究提供了强大的基线。
*   **数据集价值：** DEAL-300K 数据集规模庞大且多样化，涵盖了多种编辑场景，能够有效评估模型在不同扩散模型、不同编辑类型以及不同数据源上的泛化能力。
*   **方法优势：** MFPT 框架在各种评估场景下（包括单轮编辑、多轮编辑、真实图像检测等）均表现出优越的性能，并且在面对 JPEG 压缩和高斯模糊等图像退化时，展现出良好的鲁棒性。这表明该方法能够有效地区分扩散编辑的细微痕迹。
*   **自动化标注的潜力：** 作者提出的自动化标注流程，将人工标注时间从数千小时缩短到约42小时，为未来构建更大规模的数据集提供了可行方案。

**4. 提及的局限性：**
*   **细节精炼的不足：** 在可视化结果中提到，虽然模型能够准确地勾勒出编辑区域，但对于非常细微的细节精炼方面仍有提升空间。
*   **跨领域泛化挑战：** 虽然 DEAL-300K 旨在提高模型的跨领域泛化能力，但作者也指出，在某些特定场景下（如人脸编辑），专门训练的模型（如 Patches）可能表现更好，表明不同领域的编辑特性仍存在差异。
*   **预训练模型在不同数据集上的公平性：** 作者在评估预训练模型时提到，直接比较不同数据集上训练的模型可能存在不公平性，因为数据集的领域和特点可能存在差异。

**5. 潜在的未来研究方向：**
*   **视频篡改定位：** 作者计划将 DEAL-300K 扩展到视频领域，进一步增强其在现实世界场景中的应用性。
*   **更精细的细节定位：** 进一步优化模型，以实现更精细的编辑区域定位，捕捉更微小的篡改痕迹。
*   **更复杂的编辑场景：** 探索和构建包含多轮、多类型编辑的更复杂数据集，以应对更具挑战性的篡改场景。
*   **提升模型鲁棒性：** 继续研究模型在各种图像退化和对抗性攻击下的鲁棒性。

**总结：**

这篇论文的核心贡献在于构建了一个大规模的、专门用于扩散模型编辑区域定位的 DEAL-300K 数据集，并提出了一种创新的多频段提示微调（MFPT）框架。DEAL-300K 数据集通过创新的自动化流程生成，解决了现有数据集规模小、标注成本高的问题。MFPT 框架则通过融合视觉基础模型的强大语义理解能力和频率域的低级纹理信息，有效地捕捉了扩散模型编辑的细微特征，在多个基准测试集上取得了最先进的性能，并展现出良好的泛化能力和鲁棒性。该研究为扩散模型图像篡改的检测和定位领域奠定了坚实的基础，并为未来的相关研究提供了宝贵的资源和方法。

**Key Findings:**

- We present Diffusion-Based Image Editing Area Localization Dataset (DEAL-300K), a large scale dataset for diffusion-based image manipulation localization (DIML) with more than 300,000 annotated images.
- On top of this dataset, we propose a localization framework that uses a frozen Visual Foundation Model (VFM) together with Multi Frequency Prompt Tuning (MFPT) to capture both semantic and frequency-domain cues of edited regions.
- Trained on DEAL-300K, our method reaches a pixel-level F1 score of 82.56% on our test split and 80.97% on the external CoCoGlide benchmark, providing strong baselines and a practical foundation for future DIML research.The dataset can be accessed via https://github.com/ymhzyj/DEAL-300K.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23377v1)
- [arXiv](https://arxiv.org/abs/2511.23377v1)

---

<a id='2511.23375v1'></a>
## [Optimizing Multimodal Language Models through Attention-based Interpretability](https://arxiv.org/abs/2511.23375v1)

**Authors:** Alexander Sergeev, Evgeny Kotelnikov

**Published:** 2025-11-28

**Categories:** cs.CL, cs.CV

**Abstract:**

Modern large language models become multimodal, analyzing various data formats like text and images. While fine-tuning is effective for adapting these multimodal language models (MLMs) to downstream tasks, full fine-tuning is computationally expensive. Parameter-Efficient Fine-Tuning (PEFT) methods address this by training only a small portion of model weights. However, MLMs are difficult to interpret, making it challenging to identify which components are most effective for training to balance efficiency and performance. We propose an attention-based interpretability method for MLMs by analyzing attention scores relative to image tokens. The core idea is to identify attention heads that focus on image key objects. We utilize this information to select optimal model components for PEFT in multimodal models. Our contributions include a method for identifying attention heads associated with image key objects, its application to PEFT for image captioning, and the creation of a new dataset containing images, key object masks, and their textual descriptions. We conducted experiments on MLMs with 2-3 billion parameters to validate the method's effectiveness. By calculating Head Impact (HI) scores we quantify an attention head's focus on key objects, indicating its significance in image understanding. Our fine-tuning experiments demonstrate that adapting layers with the highest HI scores leads to the most significant shifts in metrics compared to pre-trained, randomly selected, or lowest-HI-score layers. This indicates that fine-tuning a small percentage (around 0.01%) of parameters in these crucial layers can substantially influence image understanding capabilities.

**Analysis:**

好的，这是对论文“Optimizing Multimodal Language Models through Attention-based Interpretability”的全面中文摘要，重点关注其在计算机视觉领域的创新性和重要性：

**论文题目：** 通过基于注意力的可解释性优化多模态语言模型

**作者：** Alexander Sergeev, Evgeny Kotelnikov

**摘要：**

**1. 研究问题/核心挑战：**

现代多模态语言模型（MLMs）能够同时处理文本和图像，在各种下游任务中表现出色。然而，对这些模型进行完全微调（full fine-tuning）计算成本高昂且效率低下。参数高效微调（PEFT）方法通过仅训练模型的一小部分权重来解决这个问题。但MLMs的“黑箱”特性使得理解哪些模型组件对提升效率和性能最有效变得困难，尤其是在平衡视觉和语言信息时。因此，研究的核心问题是如何**有效地识别和利用MLMs中对图像理解至关重要的组件，以便进行高效的微调**。

**2. 主要创新/方法贡献：**

该论文提出了一种**基于注意力分数的可解释性方法**，用于分析MLMs中注意力头（attention heads）对图像中关键对象（key objects）的关注程度。其核心创新点包括：

*   **注意力头关键对象关注度量（Head Impact - HI）：** 提出了一种量化方法，通过计算注意力分数与图像关键对象掩码（mask）的交并比（IoU）来衡量每个注意力头对图像关键对象的关注程度。HI分数越高，表示该注意力头越关注图像中的重要视觉元素。
*   **关键层识别与PEFT应用：** 利用HI分数识别出对图像理解贡献最大的注意力头所在的层。然后，将这些高HI分数的层作为PEFT的优先选择对象，以实现更高效、更有效的模型适应。
*   **新数据集的创建：** 构建了一个包含图像、关键对象掩码及其文本描述的新数据集，为研究和实验提供了基础。

**3. 主要结果与意义：**

通过在2-3十亿参数规模的MLMs上进行实验，研究取得了以下重要结果：

*   **高HI分数层的重要性：** 实验证明，对具有最高HI分数的层进行微调，能够带来比随机选择层或最低HI分数层更显著的模型性能提升。这表明这些层在图像理解中扮演着关键角色。
*   **PEFT的有效性：** 仅微调约0.01%的参数（针对高HI分数的层）就能显著影响模型的图像理解能力，验证了所提出方法的有效性和PEFT的潜力。
*   **可解释性与效率的结合：** 该方法成功地将模型的可解释性（识别关键组件）与模型优化（PEFT）相结合，为理解和改进MLMs提供了一条新途径。

**4. 提及的局限性：**

*   **模型架构限制：** 实验主要集中在具有相似架构（Vision Transformer编码器和Transformer解码器）的模型上，并且图像被表示为嵌入到语言模型提示中的视觉令牌。
*   **模型规模限制：** 由于计算资源的限制，实验仅限于2-3十亿参数规模的模型。
*   **任务类型限制：** 实验主要集中在图像描述（Image Captioning）和封闭式视觉问答（Visual Question Answering）任务上，并未直接评估开放式文本生成任务。为了避免歧义，实验中使用了答案模板，这可能与真实开放式生成场景有所不同。

**5. 潜在的未来研究方向：**

*   **评估开放式生成任务：** 将所提出的微调方法应用于开放式生成任务，以更全面地评估其在真实世界场景中的表现。
*   **更广泛的模型和架构：** 探索该方法在不同模型架构和更大规模模型上的适用性。
*   **更细粒度的分析：** 进一步研究注意力头在不同层级和不同模型中的行为模式，以及它们如何协同工作以实现多模态理解。
*   **跨领域和跨任务的泛化性：** 验证该方法在不同领域（如医学影像、自动驾驶）和不同多模态任务上的泛化能力。

**总结：**

这篇论文为理解和优化多模态语言模型提供了一个新颖且实用的框架。通过**基于注意力分数的可解释性方法，精确地识别出对图像理解至关重要的模型组件（即高HI分数的注意力头所在的层），并将其应用于参数高效微调（PEFT）**，研究者们证明了这种方法能够以极低的计算成本实现显著的模型性能提升。这对于推动更高效、更易于理解和部署的多模态AI模型具有重要意义，尤其是在计算机视觉与自然语言处理交叉的领域。

**Key Findings:**

- We propose an attention-based interpretability method for MLMs by analyzing attention scores relative to image tokens.
- Our contributions include a method for identifying attention heads associated with image key objects, its application to PEFT for image captioning, and the creation of a new dataset containing images, key object masks, and their textual descriptions.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23375v1)
- [arXiv](https://arxiv.org/abs/2511.23375v1)

---

<a id='2511.23369v1'></a>
## [SimScale: Learning to Drive via Real-World Simulation at Scale](https://arxiv.org/abs/2511.23369v1)

**Authors:** Haochen Tian, Tianyu Li, Haochen Liu, Jiazhi Yang, Yihang Qiu, Guang Li, Junli Wang, Yinfeng Gao, Zhang Zhang, Liang Wang, Hangjun Ye, Tieniu Tan, Long Chen, Hongyang Li

**Published:** 2025-11-28

**Categories:** cs.CV, cs.RO

**Abstract:**

Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple co-training strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +6.8 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Our simulation data and code would be released.

**Analysis:**

好的，这是对论文“SimScale: Learning to Drive via Real-World Simulation at Scale”的全面中文摘要：

**论文题目：** SimScale: Learning to Drive via Real-World Simulation at Scale

**作者：** Haochen Tian, Tianyu Li, Haochen Liu, Jiazhi Yang, Yihang Qiu, Guang Li, Junli Wang, Yinfeng Gao, Zhang Zhang, Liang Wang, Hangjun Ye, Tieniu Tan, Long Chen, Hongyang Li

**摘要：**

**1. 研究问题/核心挑战：**
实现完全自动驾驶系统需要学习在各种场景下做出理性决策，包括安全关键和分布外（out-of-distribution, OOD）的场景。然而，这些场景在人类专家收集的真实世界数据集中代表性不足。仅依赖真实世界数据进行训练，自动驾驶模型在处理罕见或未见过的情况时会遇到泛化能力不足的问题。

**2. 主要创新与方法贡献：**
为了解决数据多样性不足的问题，作者提出了一个名为 **SimScale** 的新颖且可扩展的**模拟框架**。其核心创新点包括：

*   **大规模合成未见场景：** 利用现有的驾驶日志，通过**扰动**（perturbation）的方式生成大量未见的OOD场景。
*   **高保真神经渲染：** 采用先进的**神经渲染**技术（基于3DGS [39]）和**反应式环境**（reactive environment [70]），生成高保真的多视角观测，并使其他车辆能够响应式地与自主车辆互动，从而提高模拟的真实感和多样性。
*   **伪专家轨迹生成：** 开发了一种**伪专家轨迹生成机制**，为新合成的模拟场景提供动作监督。作者比较了两种伪专家策略：
    *   **恢复式专家（Recovery-based Expert）：** 旨在将轨迹引导回人类轨迹流形内，产生人类化但保守的行为。
    *   **规划器式专家（Planner-based Expert）：** 利用特权规划器（privileged planner）生成最优轨迹，代表一种探索性策略，具有较低的真实感但更强的多样性。
*   **Sim-Real Co-training 策略：** 提出了一种简单有效的**Sim-Real Co-training**（模拟-真实联合训练）策略，将真实世界数据与合成的模拟数据结合起来训练端到端规划器。该策略旨在保持人类驾驶分布的同时，减轻模拟数据可能带来的视觉域退化问题。
*   **可扩展的数据生成流程：** SimScale框架能够通过逐步增加非重叠的模拟样本来扩展总训练数据量，同时保持真实世界数据的固定量，从而研究数据扩展的趋势。

**3. 主要结果与意义：**
通过在两个具有挑战性的真实世界基准（navhard 和 navtest）上进行广泛实验，SimScale展现了显著的优势：

*   **显著的性能提升：** Sim-Real Co-training 策略能够为多种规划方法带来**鲁棒性（robustness）和泛化能力（generalization）的协同提升**。在navhard基准上，性能最高可提升 **+6.8 EPDMS**，在navtest上可提升 **+2.9 EPDMS**。
*   **可预测的数据扩展趋势：** SimScale系统能够**清晰且可预测地扩展**，即使在真实世界数据量固定的情况下，增加模拟数据也能带来平滑的策略改进。
*   **伪专家设计的重要性：** 实验表明，**探索性更强的伪专家（如规划器式专家）比保守的恢复式专家更能带来持续的性能提升**，尤其是在数据量增加时。
*   **多模态模型优势：** 具有多模态建模能力（如DiffusionDrive）的规划器比单模态回归模型（如LTF）在数据扩展方面表现出**更强的潜力**。
*   **反应式模拟的价值：** 反应式环境下的模拟数据比非反应式模拟更能**提升真实感和多样性**，从而带来更显著的性能增益。
*   **奖励信号的有效性：** 对于评分式规划器，仅使用奖励信号进行训练（无伪专家轨迹）也能取得良好的效果，表明奖励信号在提供优化方向方面的重要性。

SimScale的成果表明，通过大规模、高保真的模拟数据，可以有效弥补真实世界数据的不足，显著提升自动驾驶系统的鲁棒性和泛化能力，为实现更安全、更可靠的自动驾驶提供了新的途径。

**4. 论文中提到的局限性：**
*   **伪专家轨迹的静态性：** 当前的伪专家轨迹扰动是静态的，作者提出未来可以探索**自演化方法**（self-evolving approach）来生成更具动态性和探索性的数据。
*   **特权规划器的局限性：** 文中使用的特权规划器是基于规则的，性能有限，可能导致舒适度指标（HC, EC）的下降，并且在极端情况下可能失效。更先进的**基于学习的特权规划器**可以改进生成效率和真实感。
*   **反应式环境的简化：** 交通行为模拟中的其他智能体由IDM [70] 控制，这限制了场景的多样性。作者建议可以采用**扩散模型**等方法来生成更丰富的交通行为。
*   **传感器模拟的局限性：** 尽管使用了3DGS，但模拟的视觉效果仍可能存在细微的**不一致性**，以及与真实世界在**分布上的差异**，这可能带来潜在风险。

**5. 未来研究方向：**
*   **自演化伪专家：** 利用自演化方法生成更动态、更具探索性的伪专家轨迹。
*   **先进的特权规划器：** 采用基于学习的特权规划器来提高伪专家轨迹的真实感和多样性。
*   **更丰富的交通行为模拟：** 探索使用扩散模型等技术生成更具多样性的交通场景。
*   **多模态传感器融合：** 整合LiDAR等其他传感器模态以提高模拟的全面性。
*   **在线强化学习与自玩：** 结合在线RL和自玩技术，进一步提升模型的学习效率和泛化能力。
*   **开放数据集与框架：** 作者承诺开源其模拟数据集和训练框架，以促进学术界和工业界在自动驾驶模拟领域的研究。

总而言之，SimScale通过创新的模拟数据生成和Sim-Real Co-training策略，有效解决了自动驾驶领域数据稀疏性问题，为提升模型在复杂和OOD场景下的性能提供了强有力的解决方案，并为未来的研究开辟了新的方向。

**Key Findings:**

- To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs.
- Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23369v1)
- [arXiv](https://arxiv.org/abs/2511.23369v1)

---

<a id='2511.23334v1'></a>
## [Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](https://arxiv.org/abs/2511.23334v1)

**Authors:** Yu Zhang, Jingyi Liu, Yiwei Shi, Qi Zhang, Duoqian Miao, Changwei Wang, Longbing Cao

**Published:** 2025-11-28

**Categories:** cs.CV

**Abstract:**

Visual AutoRegressive modeling (VAR) based on next-scale prediction has revitalized autoregressive visual generation. Although its full-context dependency, i.e., modeling all previous scales for next-scale prediction, facilitates more stable and comprehensive representation learning by leveraging complete information flow, the resulting computational inefficiency and substantial overhead severely hinder VAR's practicality and scalability. This motivates us to develop a new VAR model with better performance and efficiency without full-context dependency. To address this, we reformulate VAR as a non-full-context Markov process, proposing Markov-VAR. It is achieved via Markovian Scale Prediction: we treat each scale as a Markov state and introduce a sliding window that compresses certain previous scales into a compact history vector to compensate for historical information loss owing to non-full-context dependency. Integrating the history vector with the Markov state yields a representative dynamic state that evolves under a Markov process. Extensive experiments demonstrate that Markov-VAR is extremely simple yet highly effective: Compared to VAR on ImageNet, Markov-VAR reduces FID by 10.5% (256 $\times$ 256) and decreases peak memory consumption by 83.8% (1024 $\times$ 1024). We believe that Markov-VAR can serve as a foundation for future research on visual autoregressive generation and other downstream tasks.

**Analysis:**

好的，这是对论文“Markovian Scale Prediction: A New Era of Visual Autoregressive Generation”的全面摘要，重点关注其在计算机视觉和机器学习领域的贡献：

**论文题目：** Markovian Scale Prediction: A New Era of Visual Autoregressive Generation

**作者：** Yu Zhang, Jingyi Liu, Yiwei Shi, Qi Zhang, Duoqian Miao, Changwei Wang, Longbing Cao

**摘要：**

这篇论文提出了一种名为 **Markov-VAR** 的新型视觉自回归生成模型，旨在解决现有 Visual AutoRegressive (VAR) 模型在处理高分辨率图像生成时面临的计算效率低下和可扩展性差的问题。

**1. 研究问题/核心挑战：**

现有的 VAR 模型采用“全上下文依赖”的策略，即在预测当前尺度的图像特征时，会考虑所有之前的尺度信息。虽然这种方法有助于学习更稳定和全面的表示，但随着图像分辨率的增加，计算量呈平方级增长，导致训练和推理速度缓慢，严重限制了模型的实用性和可扩展性。此外，全上下文依赖还会导致跨尺度干扰，影响生成质量。

**2. 关键创新/方法贡献：**

*   **将 VAR 重新定义为非全上下文马尔可夫过程：** 作者将 VAR 的“下一个尺度预测”重新构想为“马尔可夫尺度预测”。他们将每个尺度视为一个马尔可夫状态，并假设当前状态的预测仅依赖于前一个状态，从而摆脱了对所有历史状态的依赖。
*   **提出马尔可夫尺度预测 (Markovian Scale Prediction)：** 这是 Markov-VAR 的核心机制。它将每个尺度的预测视为一个马尔可夫链中的一个状态转移。
*   **引入历史补偿机制 (History Compensation Mechanism)：** 为了弥补非全上下文依赖可能丢失的历史信息，作者设计了一个滑动窗口机制。该机制会压缩最近的几个尺度信息到一个紧凑的历史向量中，并将其与当前尺度的马尔可夫状态结合，形成一个代表性的动态状态，以增强预测能力。
*   **构建 Markov-VAR Transformer：** 论文还提出了一个专门的 Transformer 架构来处理马尔可夫尺度预测和历史补偿。

**3. 主要结果与意义：**

*   **显著提升效率：** 在 ImageNet 数据集上，Markov-VAR 在 256x256 分辨率下将 FID 分数降低了 10.5%，在 1024x1024 分辨率下将峰值内存消耗降低了 83.8%。这表明 Markov-VAR 在保持生成质量的同时，大幅提高了计算效率和可扩展性。
*   **生成质量的提升：** 实验结果表明，Markov-VAR 在生成质量上与 VAR 相当甚至更好，尤其是在某些情况下能产生更优的语义和更高的质量。
*   **参数效率：** 即使在参数量相当的情况下，Markov-VAR 也展现出更优的性能，证明了其参数效率。
*   **基础模型潜力：** 作者认为 Markov-VAR 可以作为未来视觉自回归生成和其他下游任务的基础模型。

**4. 提及的局限性：**

*   论文中提到，虽然马尔可夫尺度预测在效率上有所提升，但在某些情况下，与全上下文依赖相比，它可能无法完全保留所有历史信息，这可以通过历史补偿机制来缓解。
*   尽管论文展示了在 ImageNet 上的良好表现，但作者也指出，在更大、更高质量的数据集上训练，其生成质量可能会进一步提高。

**5. 潜在的未来研究方向：**

*   **更广泛的应用：** 作者希望 Markov-VAR 能作为基础模型，促进在各种下游任务中的研究。
*   **与其他模型的结合：** 论文提到，Markov-VAR 的性能和效率在与其他增强或加速技术结合时可能更具前景。
*   **进一步优化历史补偿机制：** 尽管滑动窗口机制有效，但可能还有更优化的方式来整合历史信息。

**总结：**

Markov-VAR 论文的核心贡献在于成功地将视觉自回归生成从“全上下文依赖”的范式转变为更高效的“马尔可夫尺度预测”范式。通过引入历史补偿机制，它在显著降低计算成本和内存占用的同时，保持甚至提升了生成质量。这项工作为解决高分辨率视觉生成的可扩展性问题提供了一个有前景的解决方案，并有望成为未来相关研究的重要基础。

**Key Findings:**

- This motivates us to develop a new VAR model with better performance and efficiency without full-context dependency.

**Links:**

- [PDF](https://arxiv.org/pdf/2511.23334v1)
- [arXiv](https://arxiv.org/abs/2511.23334v1)

---

