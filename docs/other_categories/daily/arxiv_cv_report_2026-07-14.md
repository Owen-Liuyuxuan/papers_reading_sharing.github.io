time: 20260714

# Arxiv Computer Vision Papers - 2026-07-14

## Executive Summary

# 每日报告执行摘要：2026-07-13 Arxiv 计算机视觉论文

## 一、主要主题与趋势

本期10篇论文呈现以下核心趋势：

- **具身智能与机器人**占据主导地位（6/10篇），涵盖世界模型、双臂移动操作、灵巧操作、导航、动力学建模等子方向。这反映出领域正从纯视觉理解向“视觉+行动”闭环快速演进。
- **视觉-语言模型（VLM/MLLM）** 继续作为核心技术组件，被用于奖励建模、视频问答、工具调用评估、导航等任务。
- **无需人类示范的学习**成为新热点：AutoPath（论文7）在导航中完全规避人工数据依赖；灵巧操作（论文5）通过重定向+强化学习简化流程。
- **单目几何与空间理解**（论文10）结合像素级场学习，推动度量几何从有监督向更通用范式发展。

## 二、重点创新论文

| 论文 | 创新点 | 意义 |
|------|--------|------|
| **3. Read It Back**（Huang等） | 提出将预训练MLLM作为**零样本奖励模型**用于文生图评估，无需额外训练即可精确对齐文本-图像一致性 | 开辟了MLLM作为通用奖励模型的新范式，可能替代人工反馈或训练好的奖励模型 |
| **2. Xiaomi-Robotics-U0**（Li等） | 提出**世界基础模型**统一多种具身合成任务（生成、规划、仿真），实现跨场景泛化 | 具身智能领域的“基础模型”尝试，有望简化机器人训练流程 |
| **7. AutoPath**（Zhang等） | 学习**可迁移的、以目标为条件的随机路径先验**，实现无需人类示范的城市级安全导航 | 打破导航对大规模人工轨迹数据的依赖，提升安全性 |
| **10. FoundationGeo**（Liu等） | 提出**空间像素级场**学习范式，单张RGB图像即可输出度量级深度、法向、曲率等几何属性 | 为单目几何估计提供统一、可扩展的框架，潜在替代传统多任务学习 |

## 三、新兴研究方向

1. **MLLM作为通用奖励/评估模型**：论文3表明，无需微调的MLLM能对生成图像进行细粒度文本对齐评估。这种方法可推广至视频、3D等多种模态生成任务。

2. **世界模型与具身基础模型**：论文2代表的世界模型方向正在从“预测未来帧”升级为支持生成、规划、仿真的统一框架，类似VLM在视觉-语言领域的覆盖度。

3. **无人类示范的自主技能学习**：论文5和7分别从灵巧操作和导航两个角度，探索如何利用物理先验（重定向、随机路径）替代昂贵的人工示教，降低机器人学习门槛。

4. **神经驱动建模与环境感知融合**：论文8将机器人动力学（关节力矩）与外力感知统一建模，这为软体机器人、接触交互任务提供了新思路。

5. **方向感知的大规模VLM导航**：论文9在城市级导航中引入方向显式编码，对比传统“图像+语言”方式，强化空间定向能力，是具身导航的前沿方向。

## 四、建议优先全文阅读的论文

- **论文3 “Read It Back”**：方法简单有效，且具有跨任务迁移潜力，适合所有从事生成评估、RLHF相关工作的研究者。
- **论文7 “AutoPath”**：放弃人类示范的设计极具前瞻性，其路径先验学习框架可对导航、运动规划领域产生直接影响。
- **论文10 “FoundationGeo”**：单目度量几何的范式转换，代码若开源将是3D视觉工作者的必备工具。
- **论文5 “A Minimalist Retargeting-Guided RL Recipe”**：灵巧操作领域的“实用手册”，对想用低成本方法解决复杂操作的实验室极具参考价值。
- **论文2 “Xiaomi-Robotics-U0”**：代表了工业界（小米）在具身基础模型上的野心，即使篇幅较长也值得了解其架构设计。

---

**总结**：本期论文显示出计算机视觉正与机器人学、具身智能深度融合，而MLLM作为“元模型”的角色愈发突出。无人类示范学习和世界基础模型是未来数月值得持续跟踪的两个关键方向。

---

## Table of Contents

1. [Evidence-Backed Video Question Answering](#2607.11862v1)
2. [Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model](#2607.11643v1)
3. [Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation](#2607.11886v1)
4. [Mixture of Frames Policy: Multi-Frame Action Denoising for Bimanual Mobile Manipulation](#2607.11884v1)
5. [A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation](#2607.11874v1)
6. [MM-ToolSandBox: A Unified Framework for Evaluating Visual Tool-Calling Agents](#2607.11818v1)
7. [AutoPath: Learning Transferable Goal-Conditioned Stochastic Path Prior for Safe Navigation Without Human Demonstrations](#2607.11739v1)
8. [NeuralActuator: Neural Actuation Modeling for Robot Dynamics and External Force Perception](#2607.11734v1)
9. [DA-Nav: Direction-Aware City-Scale Vision-Language Navigation](#2607.11638v1)
10. [FoundationGeo: Learning Spatial Pixel-Wise Fields for Monocular Metric Geometry](#2607.11588v1)

---

## Papers

<a id='2607.11862v1'></a>
## [Evidence-Backed Video Question Answering](https://arxiv.org/abs/2607.11862v1)

**Authors:** Shijie Wang, Honglu Zhou, Ziyang Wang, Ran Xu, Caiming Xiong, Silvio Savarese, Chen Sun, Juan Carlos Niebles

**Published:** 2026-07-13

**Categories:** cs.CV, cs.AI

**Abstract:**

Current Video Large Language Models (Video LLMs) excel in question answering (QA) but largely operate as black boxes, providing textual answers without verifiable visual grounding. Existing explainability efforts rely on textual rationales or sparse bounding boxes, which struggle to capture complex video dynamics such as occlusions and non-rigid deformations. We propose Evidence-Backed Video Question Answering (E-VQA), a novel task requiring models to jointly output a semantic answer and precise spatio-temporal evidence: temporal segments and dense, tracked object segmentation masklets. To support this, we introduce ST-Evidence, the first human-verified benchmark for both discriminative and generative pixel-level grounding. Evaluations of state-of-the-art models reveal a critical decoupling between QA accuracy and true visual perception that scaling alone fails to bridge. To address this, we develop scalable, automated generation pipelines to create ST-Evidence-Instruct, a 160k-scale dataset bridging high-level reasoning with fine-grained grounding. Fine-tuning grounded Video LLMs on this data yields substantial gains over the corresponding size-matched UniPixel baselines (e.g., +27.2 t-mean and +13.8 J&F on a 7B model), establishing a robust baseline for explainable, evidence-backed video understanding. Code and data are available at https://github.com/SalesforceAIResearch/EVQA.

**Analysis:**

以下是对《Evidence-Backed Video Question Answering》论文的方法论深度分析：

### 1. 摘要翻译
当前视频大语言模型（Video LLMs）在问答（QA）任务中表现出色，但往往以黑盒形式运行，仅提供文本答案，缺乏可验证的视觉基础。现有的可解释性研究多依赖文本解释或稀疏边界框，难以捕捉遮挡、非刚性形变等复杂视频动态。我们提出了**证据支持的视频问答（E-VQA）**，这是一项要求模型联合输出语义答案与精确时空证据（即时间段及密集跟踪的目标分割掩码/Masklets）的任务。为支持该任务，我们引入了**ST-Evidence**，这是首个用于判别式和生成式像素级基础的经人工验证的基准。评估显示，模型QA准确率与真实视觉感知之间存在关键脱钩。为此，我们开发了可扩展的自动化生成流水线，构建了包含160k条数据的**ST-Evidence-Instruct**数据集，实现了高层推理与细粒度基础的桥接。实验证明，在该数据集上进行指令微调显著提升了视频LLM的联合推理与像素级基础能力。

### 2. 方法动机分析
*   **驱动力**：解决Video LLM在处理高风险领域（如自动驾驶、医疗分析）时缺乏可解释性、容易产生“幻觉”的问题，将语言模型对视频的理解绑定到像素级的“视觉证明”上。
*   **痛点**：现有方法将“推理”与“基础”（Localization）割裂；大多仅关注稀疏定位（关键帧/边界框），无法处理复杂动态（如物体遮挡、形变）。
*   **研究假设**：通过显式要求模型输出稠密的时空分割掩码（Masklets），可以迫使模型将语义推理“锚定”在真实的视觉线索上，从而消除语言先验偏差。

### 3. 方法设计详解
E-VQA任务要求生成三元组 $(A, E_t, E_s)$：答案、时间段、时空掩码。
*   **自动化流水线（ST-Evidence-Instruct）**：
    *   **路径1（已知掩码数据）**：利用ViCaS数据集的既有掩码，通过两个VLM（Qwen3-VL和Gemini）协同生成问题与答案，经由置信度筛选及文本审查，自动对齐QA与mask。
    *   **路径2（仅有QA数据）**：提出“分解Pipeline”：
        1.  **证据识别**：引导VLM识别支持答案的核心对象（Referring expressions）及关键时间戳。
        2.  **边界框生成**：基于描述生成初始框，滤除高退化预测。
        3.  **密集掩码传播**：使用SAM-3模型，以边界框为提示，在全视频空间中进行像素级掩码传播，并去重。
*   **模型结构**：基于UniPixel架构，利用LoRA高效微调视觉编码器与LLM，同时完整训练掩码解码器。损失函数由下一token预测（QA+时序）与掩码解码损失（Masklets）构成。

### 4. 方法对比分析
*   **本质区别**：从传统的“QA+辅助提示”转变为“QA+像素级证据证明”，将Masklets作为最终回答的必要组成部分。
*   **创新点**：首个要求输出密集时空Masklets的基准；自动化构建高质量海量指令微调数据集的Pipeline（利用VLM+SAM-2/3实现弱监督到强监督的转化）。
*   **适用场景**：需要高置信度、可解释性极强的视频理解系统。

### 5. 实验分析
*   **验证方法**：在ST-Evidence-Gen（生成）与ST-Evidence-MCQ（判别）上评估。
*   **关键结论**：缩放模型参数（Scaling）无法解决视觉基础难题，数据质量与任务设定才是核心；通过指令微调，Ours-7B在分割精度（J&F）上大幅超越同规模基线，且不损失泛化能力。
*   **局限**：对长视频的密集处理计算代价较高；依赖代理模型进行掩码评估时存在语义-视觉对齐瓶颈。

### 6. 实用指南
*   **开源**：代码与数据集已发布（https://github.com/SalesforceAIResearch/EVQA）。
*   **实现细节**：微调时需混合多种视频理解数据以维持泛化性；SAM-3作为掩码传播核心，确保时序连续性是关键。
*   **迁移**：该Pipeline可迁移至任何需要特定视觉对象定位的视频理解任务（如动作识别、异常检测）。

### 7. 总结
*   **核心思想**：用像素级证据链锚定语义推理，实现可解释的视频大模型。
*   **速记版pipeline**：
    1.  筛选视频及问答对。
    2.  大模型识别核心证据对象与时间段。
    3.  利用视觉分割模型生成密集Masklets。
    4.  联合微调大模型以对齐语义与时空像素。

**Key Findings:**

- We propose Evidence-Backed Video Question Answering (E-VQA), a novel task requiring models to jointly output a semantic answer and precise spatio-temporal evidence: temporal segments and dense, tracked object segmentation masklets.
- To support this, we introduce ST-Evidence, the first human-verified benchmark for both discriminative and generative pixel-level grounding.
- Evaluations of state-of-the-art models reveal a critical decoupling between QA accuracy and true visual perception that scaling alone fails to bridge.
- To address this, we develop scalable, automated generation pipelines to create ST-Evidence-Instruct, a 160k-scale dataset bridging high-level reasoning with fine-grained grounding.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11862v1)
- [arXiv](https://arxiv.org/abs/2607.11862v1)

---

<a id='2607.11643v1'></a>
## [Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model](https://arxiv.org/abs/2607.11643v1)

**Authors:** Xinghang Li, Jun Guo, Qiwei Li, Long Qian, Hang Lai, Yueze Wang, Hongyu Yan, Jiahang Cao, Xi Chen, Jingen Qu, Jiaxi Song, Nan Sun, Hanye Zhao, Futeng Liu, Wanli Peng, Heyun Wang, Yunhong Wang, Caoyu Xia, Jack Zhao, Diyun Xiang, Hangjun Ye, Heng Qu, Huaping Liu, Jason Li

**Published:** 2026-07-13

**Categories:** cs.RO, cs.AI

**Abstract:**

Recent foundation image and video generation models offer strong generalization and controllability, but their direct application to embodied scenarios is limited by requirements for multi-view consistency, geometric coherence, and robot embodiment constraints. Existing methods typically adapt foundation models with limited robot data, often sacrificing visual knowledge acquired during large-scale pre-training. We present Xiaomi-Robotics-U0, a 38-billion-parameter multimodal autoregressive model for unified embodied synthesis. It treats embodied generation as an extension of foundation image and video generation and jointly optimizes text-to-image generation, image editing, embodied scene generation, embodied transfer, and embodied video generation. This unified framework preserves the generalization of the pre-trained world foundation model while adapting it to embodied settings. Xiaomi-Robotics-U0 is the first model to support high-quality multi-view scene generation across multiple robot embodiments and to introduce structured, controllable embodied transfer for fine-grained editing while preserving multi-view consistency and interaction dynamics. It achieves state-of-the-art results on single-step and sequential generation tasks, outperforming GPT-Image-2.0 in human evaluations of embodied scene generation and transfer, ranking first on World Arena for embodied video generation, and improving the out-of-distribution success rate of pi_0.5 from 36.9% to 63.2% on challenging real-world manipulation tasks. These results show that foundation world models can serve both as embodied world models and scalable data engines for embodied intelligence. Code and checkpoints are available at https://robotics.xiaomi.com/xiaomi-robotics-u0.html.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇关于 **Xiaomi-Robotics-U0** 的论文摘要分析如下：

### 1. 核心贡献总结
Xiaomi-Robotics-U0 是一款拥有 380 亿参数的统一多模态自回归大模型，旨在将通用视觉生成能力与具身智能（Embodied AI）深度融合。该模型不仅实现了高质量的多视角具身场景生成，还通过结构化、可控的具身迁移（Embodied Transfer）技术，解决了具身智能中视觉一致性与物理交互动力学的平衡难题，为具身智能领域提供了一个强有力的生成式世界模型与数据引擎。

### 2. 关键创新与方法论
*   **统一生成框架（Unified Framework）：** 该模型打破了通用视觉生成与具身任务之间的壁垒，通过“统一具身合成”将图像生成、编辑、场景生成及视频生成整合到一个自回归架构中，最大限度地保留了预训练视觉大模型的基础知识。
*   **多视角与几何一致性：** 区别于以往仅对基础模型进行微调的做法，U0 重点解决了具身任务中至关重要的多视角一致性（Multi-view Consistency）和几何相干性，确保生成的场景符合真实物理约束。
*   **结构化具身迁移（Structured Embodied Transfer）：** 创新性地引入了具身迁移机制，允许在进行精细化编辑的同时，保持交互动力学的连贯性，这是实现高效复杂机器人操作模拟的关键。
*   **数据引擎角色：** 模型不仅是一个模拟器，更是一个大规模高质量合成数据引擎，通过增强分布外（OOD）任务的成功率，证明了其作为机器人策略学习“助推器”的潜力。

### 3. 对该领域的潜在影响
*   **具身智能的“数据饥渴”解决方案：** 该研究展示了世界模型可以通过合成高质量数据，极大地提升机器人策略在分布外环境中的表现（从 36.9% 提升至 63.2%），这为解决具身领域数据匮乏的问题指明了明确路径。
*   **重塑具身模拟：** 随着 U0 在 World Arena 等基准测试中的优异表现，它可能改变传统的仿真器（如 MuJoCo 或 Isaac Sim）在机器人训练中的地位，转向基于生成式 AI 的“神经仿真”。
*   **多模态模型范式转移：** 将具身任务建模为自回归生成任务，进一步巩固了生成式世界模型（World Foundation Models）在构建通用机器人智能中的核心地位。

### 4. 相关领域与受益应用
*   **自动驾驶与移动机器人：** 需要高保真、长时序场景预测，用于感知训练或路径规划。
*   **数字孪生（Digital Twins）：** 能够根据简单文本指令快速构建与物理规则一致的复杂交互环境。
*   **人机交互（HCI）：** 通过精细化编辑技术，实现对机器人工作空间的实时动态调整，提升人机协作的安全性与灵活性。
*   **视频生成与虚拟制作：** U0 强大的多视角控制能力可直接迁移到影视后期合成或 VR/AR 内容创作中。

### 5. 可推断的局限性
*   **实时性挑战：** 380 亿参数模型在处理推理延迟方面往往存在挑战，对于需要毫秒级响应的实时机器人控制任务，其计算效率可能需要进一步优化（如蒸馏或硬件加速）。
*   **长时序因果稳定性：** 尽管摘要强调了交互动力学，但在极长序列的交互中，纯自回归模型通常仍面临“漂移（Drift）”问题，即随着时间推移，物理一致性可能随之下降。
*   **对物理法则的隐式建模能力：** 尽管模型效果卓越，但作为一种基于视觉的模型，它对底层复杂物理交互的理解（例如接触力、摩擦力等）是否完全等同于显式物理引擎，仍值得在更复杂的极端环境下验证。

**总结评价：** 
Xiaomi-Robotics-U0 的重要性在于它**成功地将“视觉创造力”转化为了“具身控制力”**。它证明了通用视觉大模型并非只能生成静态图片或炫酷视频，而是可以成为理解物理世界、支撑具身决策的高级认知工具。这对目前处于“scaling data”阶段的具身智能研究具有极高的战略价值。

**Key Findings:**

- We present Xiaomi-Robotics-U0, a 38-billion-parameter multimodal autoregressive model for unified embodied synthesis.
- It achieves state-of-the-art results on single-step and sequential generation tasks, outperforming GPT-Image-2.0 in human evaluations of embodied scene generation and transfer, ranking first on World Arena for embodied video generation, and improving the out-of-distribution success rate of pi_0.5 from 36.9% to 63.2% on challenging real-world manipulation tasks.
- These results show that foundation world models can serve both as embodied world models and scalable data engines for embodied intelligence.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11643v1)
- [arXiv](https://arxiv.org/abs/2607.11643v1)

---

<a id='2607.11886v1'></a>
## [Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation](https://arxiv.org/abs/2607.11886v1)

**Authors:** Runhui Huang, Qihui Zhang, Zhe Liu, Yu Gao, Jie Wu, Hengshuang Zhao

**Published:** 2026-07-13

**Categories:** cs.CV

**Abstract:**

In this paper, we propose SpectraReward, a training-free reward function that turns pretrained MLLMs into off-the-shelf reward models for image-generation reinforcement learning. Instead of asking the MLLM to judge a generated image or answer decomposed verification questions, SpectraReward measures how well the original prompt can be recovered from the generated image through a single image-conditioned, teacher-forced forward pass. We use the average image-conditioned prompt log-likelihood as the reward, directly reusing the MLLM's pretrained image-text alignment ability without preference labels, reward-model fine-tuning. We further introduce Self-SpectraReward, a special case for unified multimodal models where the policy's own understanding branch serves as the reward model for its generation branch, forming a closed-loop self-improving framework without external reward models or external knowledge. Extensive experiments validate SpectraReward through a broad image-generation RL study covering two diffusion models, three RL algorithms, nine reward MLLM backbones from four MLLM families spanning 4B to 235B parameters, and five out-of-distribution text-to-image benchmarks. Results show that both SpectraReward and Self-SpectraReward significantly and consistently improve generation performance and outperform prior MLLM-derived reward training methods. Further analysis reveals that larger reward MLLMs are not always better, while Self-SpectraReward can match or surpass much larger external reward models, suggesting that reward-policy alignment is a key factor for effective image-generation RL. Project Page: https://huangrh99.github.io/SpectraReward/

**Analysis:**

这份报告对《Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation》进行了深度分析。

### 1. 摘要翻译
本文提出了 **SpectraReward**，一种无需训练的奖励函数，能将预训练的多模态大模型（MLLM）转化为即插即用的文本到图像生成强化学习奖励模型。SpectraReward 不要求 MLLM 对图像进行打分或回答问题，而是通过单次教师强制（teacher-forced）的前向传递，测量生成的图像能多好地“还原”原始提示词（prompt）。通过使用图像条件下的平均提示词对数似然作为奖励，该方法直接复用了 MLLM 的预训练图像-文本对齐能力。此外，作者引入了 **Self-SpectraReward**，让统一多模态模型（UMM）的理解分支为其生成分支提供奖励，形成了无须外部奖励模型的闭环自改进框架。广泛实验证明，该方法显著提升了图像生成性能，且在模型规模不变的情况下，表现优于许多大得多的外部奖励模型。

### 2. 方法动机分析
- **驱动力**：解决图像生成RL中奖励模型设计成本高、依赖人类偏好标注或复杂问答流水线的问题。
- **现有方法痛点**：
    - **标注依赖**：基于人类偏好的奖励模型训练昂贵且缓慢。
    - **鲁棒性差**：直接让MLLM输出标量分数（Scalar Score）极易受到校准偏置和评分噪声影响。
    - **工程复杂度**：基于问题拆解（VQA）的方案涉及繁琐的流水线，难以大规模部署。
- **研究假设**：提示词在图像条件下的可预测性（对数似然）是评估图文对齐程度的有效且鲁棒的内在度量，无需额外的监督数据。

### 3. 方法设计详解
- **核心 pipeline**：
    1. **输入**：生成图像 $y$ 和原始提示词 $x = (x_1, \dots, x_T)$。
    2. **教师强制前向传递**：将图像 $y$ 作为 visual condition 输入到冻结的 MLLM 中。
    3. **似然计算**：计算提示词中每个 token 在图像条件下的条件概率 $p(x_{t+1} | x_{\leq t}, y)$。
    4. **语义频谱聚合**：计算整个提示词的平均对数似然 $R_M(x, y) = \frac{1}{T-1} \sum_{t=1}^{T-1} \log p_M(x_{t+1} | x_{\leq t}, y)$。
    5. **奖励优化**：该标量 reward 直接参与 RL 更新。
- **Self-SpectraReward**：当生成器本身是统一多模态模型（如 BAGEL）时，直接调用其内部的“理解分支”作为奖励模型。由于两者共享 Tokenizer、视觉编码器和预训练分布，这种“奖励-策略对齐”最大化了奖励信号的可靠性。

### 4. 方法对比分析
- **本质区别**：从“让模型做判断（Judge）”转变为“评估模型对提示词的重构能力（Reconstruction/Prediction）”。
- **创新贡献**：
    - **训练零门槛**：无需任何偏好数据，利用模型预训练的内在概率分布。
    - **闭环自我改进**：通过 Self-SpectraReward 消除了对外部大型奖励模型的依赖。
    - **对齐效率**：论证了分布对齐（Distributional Alignment）在强化学习中的作用可能超过单纯的模型参数规模。

### 5. 实验分析
- **关键结论**：在多个 benchmark（如 TIIF-Bench, GenEval）上，SpectraReward 显著优于强基线 AlphaGRPO。Self-SpectraReward 的表现可媲美甚至超越 30 倍参数规模的外部奖励模型。
- **局限性**：依赖 MLLM 的视觉推理能力上限；对于极其隐式的物理/常识要求（如“热咖啡”隐含的热气），纯文本提示词似然可能响应较弱。

### 6. 实用指南
- **开源/复现**：项目主页 `https://huangrh99.github.io/SpectraReward/`。
- **关键细节**：
    - **Mask EOS**：计算时需忽略 [EOS] token，因为它通常受序列终止影响，会引入噪声。
    - **Prefix 策略**：多数模型无须加 prefix，InternVL3.5 建议添加“Describe the image”以提高稳定性。
    - **迁移建议**：该方法非常适合任何具有统一理解与生成能力的 UMM 架构，仅需提取其 Cross-Attention 或视觉-文本建模接口即可。

### 7. 总结
- **核心思想**：将图像作为条件，评估提示词在多模态大模型下的条件概率，作为评估对齐程度的天然奖励信号。
- **速记版pipeline**：
    1. 输入提示词和待评分图像。
    2. 使用图像作为视觉输入，对提示词进行单次前向推理。
    3. 计算每个提示词 Token 的似然均值。
    4. 将该均值作为 RL 的标量奖励进行训练。

**Key Findings:**

- In this paper, we propose SpectraReward, a training-free reward function that turns pretrained MLLMs into off-the-shelf reward models for image-generation reinforcement learning.
- Results show that both SpectraReward and Self-SpectraReward significantly and consistently improve generation performance and outperform prior MLLM-derived reward training methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11886v1)
- [arXiv](https://arxiv.org/abs/2607.11886v1)

---

<a id='2607.11884v1'></a>
## [Mixture of Frames Policy: Multi-Frame Action Denoising for Bimanual Mobile Manipulation](https://arxiv.org/abs/2607.11884v1)

**Authors:** Dian Wang, Jisang Park, Xiaomeng Xu, Han Zhang, Shuran Song, Jeannette Bohg

**Published:** 2026-07-13

**Categories:** cs.RO

**Abstract:**

Robotic manipulation is inherently multi-frame: local actions may be simple in an end-effector frame, while transport, upright-object handling, and whole-body coordination are better represented in a base-aligned frame. However, modern diffusion-based visuomotor policies typically commit to a single predefined action frame, forcing one denoiser to model action distributions that are often unnecessarily complex in that frame. We propose Mixture of Frames Policy (MoF), a diffusion policy that performs synchronized action denoising across multiple coordinate frames. MoF maintains a single canonical diffusion state, re-expresses it in several task-relevant frames, applies frame-specialized denoisers, and fuses their noise predictions back in the canonical frame. To make this possible for intermediate noisy diffusion states, we introduce a column-based 6D rotation representation within an SE(3) action parameterization that supports exact, differentiable frame transformations without requiring noisy rotations to lie on the SO(3) manifold. Across nine simulated bimanual manipulation tasks, we show that the best action frame is task-dependent and that MoF improves over oracle frame selection and standard Mixture-of-Experts (MoE) baselines. We further evaluate MoF on two real-world bimanual mobile manipulation tasks, demonstrating that it outperforms all constituent single-frame baselines. Project homepage: https://mofpo.github.io

**Analysis:**

### 1. 摘要翻译
机器人操作本质上是多坐标系下的任务：局部动作在末端执行器坐标系中表现简单，而物体运输、直立搬运及全身协调则在基座对齐坐标系中更易建模。然而，现代基于扩散模型的视觉运动策略通常仅使用单一预定义的动作坐标系，导致去噪器在处理复杂动作分布时面临不必要的难度。我们提出了“坐标系混合策略”（Mixture of Frames Policy, MoF），这是一种在多个坐标系中同步进行动作去噪的扩散策略。MoF维持单一的规范化扩散状态，将其表达在多个任务相关坐标系中，应用各坐标系专有的去噪器，最后将噪声预测融合回规范坐标系。为此，我们引入了一种基于列向量的6D旋转表示法，支持在SE(3)空间内进行精确、可微分的坐标系变换，且无需将带有噪声的旋转强制投影到SO(3)流形上。在九项模拟及两项现实世界双臂移动操作任务中，MoF均优于单坐标系策略及基线方法。

### 2. 方法动机分析
- **驱动力**：操作任务的复杂性往往随坐标系选择而剧烈波动。单一预定义坐标系无法在整个任务生命周期内保持最优性。
- **现有方法痛点**：当前策略（如Diffusion Policy）强行将动作分布建模在单一坐标系内，忽略了部分动作在特定坐标系下具有不变性或更简单的几何结构。
- **研究假设**：通过在并行坐标系中利用“专家”进行同步去噪，并结合路由机制或融合策略，能够自动根据任务进度在不同“动作视角”间切换，从而大幅提升策略的学习效率与执行鲁棒性。

### 3. 方法设计详解
- **核心流程**：
  1. **规范化状态维护**：在规范坐标系（Canonical Frame）$F_c$中保持一个单一的扩散去噪状态 $x_c^k$。
  2. **多坐标系投影**：通过已知的刚体变换 ${}^m T_c$，将噪声状态 $x_c^k$ 分别转换至各个任务相关的专家坐标系 $F_m$。
  3. **并行专家去噪**：每个专家模型 $\epsilon_\theta^m$ 在其特有的坐标系下预测噪声，这些专家专注于建模该坐标系下最简单的动作流形。
  4. **噪声融合**：将各专家预测的噪声转换回规范空间，通过可学习的路由器（Router）权重 $w_m$ 或均匀加权，聚合为最终的噪声预测 $\hat{\epsilon}_c$。
  5. **协同去噪**：利用聚合的 $\hat{\epsilon}_c$ 执行去噪步骤，确保所有专家在同一步调下同步演化，避免模式失配。
- **关键技术创新（SE(3)列向量表达）**：为避免在去噪过程中对旋转矩阵进行非线性的正交化投影（这会破坏噪声的分布性质），作者提出将6D旋转表示为两个列向量。坐标系变换仅需左乘旋转矩阵，这在噪声状态下依然是精确且可微分的线性变换。

### 4. 方法对比分析
- **本质区别**：与MoE-DP等方法不同，MoF是将Mixture-of-Experts结构置于**去噪过程本身**，而非条件特征提取路径中。
- **创新贡献**：
  - 提出了基于动作坐标系转换的去噪架构。
  - 引入了无需投影的变换兼容旋转表示法，解决了扩散模型中间状态的几何变换难题。
- **适用场景**：适用于复杂双臂协调、多阶段移动操作任务，即任务的不同阶段对不同参考系有显著偏好的场景。

### 5. 实验分析（精简版）
- **验证方法**：在BiGym和DexMimicGen的九个任务中与单坐标系基线、Oracle基线进行对比。
- **关键结果**：MoF平均成功率显著超过单坐标系基线（如平均提升16.5%），且在Threading等任务中通过路由机制展现了符合任务物理结构的动态坐标系切换行为。
- **主要局限**：依赖于设计者预定义的坐标系集合，目前尚未实现坐标系的自动发现。

### 6. 实用指南
- **开源情况**：官方主页 [mofpo.github.io](https://mofpo.github.io) 提供详细信息。
- **实现细节**：
  - **旋转表示**：必须使用文中的列向量表示，避免在去噪步骤中进行 $SO(3)$ 投影。
  - **辅助损失**：需要增加 per-expert 辅助损失（Auxiliary Loss），防止权重较低的专家在训练中“漂移”。
- **迁移可能**：该方法极易迁移到任何涉及多参考系的机器人操作任务中，尤其是当任务需要同时关注“物体-手”相对关系和“基座-环境”空间关系时。

### 7. 总结
- **核心思想**：利用多视角动作去噪专家，实现坐标系感知的协同优化。
- **速记版pipeline**：
  1. 将当前 noisy 动作投影到各坐标系；
  2. 多位专家并行预测噪声；
  3. 将噪声变换回公共空间并加权融合；
  4. 更新扩散采样结果。

**Key Findings:**

- We propose Mixture of Frames Policy (MoF), a diffusion policy that performs synchronized action denoising across multiple coordinate frames.
- To make this possible for intermediate noisy diffusion states, we introduce a column-based 6D rotation representation within an SE(3) action parameterization that supports exact, differentiable frame transformations without requiring noisy rotations to lie on the SO(3) manifold.
- Across nine simulated bimanual manipulation tasks, we show that the best action frame is task-dependent and that MoF improves over oracle frame selection and standard Mixture-of-Experts (MoE) baselines.
- We further evaluate MoF on two real-world bimanual mobile manipulation tasks, demonstrating that it outperforms all constituent single-frame baselines.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11884v1)
- [arXiv](https://arxiv.org/abs/2607.11884v1)

---

<a id='2607.11874v1'></a>
## [A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation](https://arxiv.org/abs/2607.11874v1)

**Authors:** Yunhai Feng, Natalie Leung, Jiaxuan Wang, Lujie Yang, Haozhi Qi, Preston Culbertson

**Published:** 2026-07-13

**Categories:** cs.RO, cs.AI, cs.LG

**Abstract:**

Recent work in humanoid whole-body control has found success with a simple recipe: retarget human motion to robot kinematic references, then train policies via reinforcement learning (RL) to track them. But how does this recipe transfer to dexterous manipulation? The answer is not obvious, as manipulation involves complex, contact-rich dynamics and requires delicate regulation of contact modes and forces. We present REGRIND, a minimalist retargeting-guided RL pipeline that learns dexterous manipulation policies from a single human demonstration. REGRIND retargets human hand-object motion to a robot reference that preserves hand-object spatial and contact relationships, trains a residual RL policy in simulation to track object-centric keypoints along that reference, and transfers the resulting policy zero-shot to hardware with careful system identification. The resulting policies produce fluid, human-like behavior on two different multi-fingered hands across contact-rich tool-use tasks, including operating a pair of scissors and turning a screwdriver. Through systematic hardware experiments, we identify and analyze the key factors that govern sim-to-real transfer in dexterous manipulation, offering practical guidance for retargeting-based learning in contact-rich settings. Videos and code are available at https://yunhaifeng.com/REGRIND.

**Analysis:**

以下是对该论文的深入分析：

### 1. 摘要翻译
人类全身控制领域已成功采用一种简单范式：将人类动作重定向为机器人运动参考，随后通过强化学习（RL）进行策略训练。然而，这种范式在灵巧操作中的迁移并不直观，因为操作任务涉及复杂的接触动力学及作用力调节。我们提出了REGRIND，这是一种极简的重定向引导RL流水线，仅需单次人类演示即可学习灵巧操作。REGRIND通过保持手-物空间与接触关系的交互感知重定向来生成机器人参考，利用残差RL在仿真中追踪物体中心关键点，并通过精细的系统辨识实现零样本硬件迁移。实验表明，该方法在剪刀、螺丝刀等接触丰富的灵巧操作任务上展现出流畅的人类般行为，并揭示了 sim-to-real 迁移的关键要素。

### 2. 方法动机分析
- **驱动力**：解决灵巧手操作任务中，如何利用海量人类动作数据实现高效、鲁棒的策略学习。
- **现有方法痛点**：传统的运动重定向（如简单IK）忽视了手-物交互的动力学语义，生成的轨迹物理不可行，导致RL在下游训练时缺乏高质量的指导，或者产生了错误的接触结构。
- **研究假设**：通过“交互保持（interaction-preserving）”的重定向生成高质量参考轨迹，能为RL提供正确的动作先验，从而解决sim-to-real中的接触建模敏感性问题。

### 3. 方法设计详解
REGRIND由三个核心环节构成：
1. **交互感知运动重定向（Interaction-Aware Retargeting）**：
   - 核心在于构建“交互网格（Interaction Mesh）”。作者不仅重定向手部关键点，还引入物体关键点。通过Delaunay四面体化建立手与物体的连接，优化过程中最小化重定向前后网格的拉普拉斯坐标差异，从而在机器人维度上“复刻”了人手的接触语义。
2. **基于参考的RL训练（Reference-Guided RL）**：
   - 采用残差RL策略：$q_{target} = \bar{q} + \alpha \odot \pi_{\theta}(\bar{q}, o_t)$，其中 $\bar{q}$ 是重定向轨迹。
   - 关键点在于使用重定向轨迹作为“探索重启分布（RSI）”，引导智能体在任务的关键空间区域进行探索。
3. **动态数据增强**：
   - 通过对初始物体姿态进行小幅扰动，并利用时间变化的刚体变换将该扰动平滑“回弹”至原始目标状态，从而合成多样化的参考轨迹，提升模型对不同初始状态的泛化能力。

### 4. 方法对比分析
- **本质区别**：与传统运动重定向不同，REGRIND不是单纯追求外观上的相似，而是通过交互网格保证接触点空间关系的拓扑不变性。
- **创新贡献**：提出了一种无需复杂接触标注的语义重定向框架，以及一套结合动态数据增强与参考状态初始化的RL训练方案。
- **适用场景**：适用于需要精确接触反馈、复杂交互的灵巧操作任务（如工具使用）。

### 5. 实验分析
- **验证方法**：在LEAP和WUJI两款机器人手上，测试了剪刀与螺丝刀任务。
- **关键结果**：在仿真中达到了近100%的成功率，显著优于SPIDER和DexMachina；在真实硬件上实现了跨形态的零样本迁移，且具备良好的初始状态泛化性。
- **局限**：对系统辨识（System ID）要求较高；目前依赖动作捕捉数据，尚未实现端到端的视觉输入。

### 6. 实用指南
- **开源情况**：已开源，代码见 [yunhaifeng.com/REGRIND](https://yunhaifeng.com/REGRIND)。
- **实现关键**：
  - **交互网格**：需重点关注接触点（Handles/Surface）的分布采样。
  - **训练细节**：Table 4中的奖励权重设置对收敛至关重要，特别是对动作速率和物体位置误差的权衡。
  - **迁移建议**：必须在仿真中加入与现实匹配的控制延迟（time lag）及噪声，否则sim-to-real极易失效。

### 7. 总结
- **核心思想**：通过拓扑一致的交互网格重定向，将人手的操作语义注入机器人策略训练中。
- **速记版Pipeline**：
  1. **构建网格**：提取人手与物体的连接关系。
  2. **运动重定向**：优化机器人轨迹以保持上述拓扑关系。
  3. **数据扩增**：对初始位姿随机化生成多条参考轨迹。
  4. **残差学习**：以轨迹作为引导进行仿真RL训练。
  5. **实机部署**：通过系统辨识消除仿真与现实的响应差距。

**Key Findings:**

- We present REGRIND, a minimalist retargeting-guided RL pipeline that learns dexterous manipulation policies from a single human demonstration.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11874v1)
- [arXiv](https://arxiv.org/abs/2607.11874v1)

---

<a id='2607.11818v1'></a>
## [MM-ToolSandBox: A Unified Framework for Evaluating Visual Tool-Calling Agents](https://arxiv.org/abs/2607.11818v1)

**Authors:** Kaixin Ma, Di Feng, Alexander Metz, Jiarui Lu, Eshan Verma, Afshin Dehghan

**Published:** 2026-07-13

**Categories:** cs.CV, cs.AI

**Abstract:**

We introduce MM-ToolSandBox, a benchmark and evaluation framework for visually grounded tool-calling agents. The framework provides a stateful execution environment spanning 500+ tools across 16 application domains, supporting multi-image, multi-turn tasks where agents must ground progressively arriving visual inputs into executable tool calls while handling realistic conversational phenomena (goal revisions, error corrections, state mutations). An automated scenario generation pipeline produces diverse, visually grounded scenarios through information-flow-guided planning and multi-stage quality filtering, yielding 258 human-verified nominal scenarios and 50 variants targeting interactive UI applications. Evaluating 12 state-of-the-art models, from 4B open-weight to frontier proprietary systems, shows that current models still lack robust visual tool-calling capability: even the best model achieves below 50% success rate. Our failure analysis further reveals that visual precision, not only planning, is a primary bottleneck for capable models: 53% of failures stem from incorrect information extraction from images despite otherwise correct task workflows. A planning-to-precision crossover emerges with scale: smaller models fail at deciding what to do, while larger models fail at perceiving what they see, suggesting fundamentally different research directions for improving models at different capability levels. The framework and the benchmark are publicly available at https://github.com/apple/ml-mmtoolsandbox

**Analysis:**

### 1. 摘要翻译
我们推出了 **MM-ToolSandBox**，这是一个针对视觉接地（visually grounded）工具调用代理的基准测试和评估框架。该框架提供了一个跨越 16 个应用领域、包含 500 多个工具的状态化执行环境，支持多图像、多轮对话任务。在这些任务中，代理必须将逐步到达的视觉输入接地到可执行的工具调用中，同时处理现实的对话现象（如目标修订、纠错、状态变动）。我们开发了一个自动场景生成流水线，通过信息流引导的规划和多阶段质量过滤，生成了 258 个经人工验证的标称场景和 50 个针对交互式 UI 应用的变体。对 12 个最先进模型（从 4B 开源模型到前沿专有系统）的评估显示，当前模型仍缺乏稳健的视觉工具调用能力：即使最好的模型成功率也低于 50%。我们的故障分析进一步揭示，视觉精度（而非规划）是高能力模型的主要瓶颈：53% 的故障源于从图像中提取信息错误。随着规模扩大，出现了一个“从规划到精度”的交叉现象：小模型难以决定“做什么”，而大模型难以感知“看到了什么”，这表明提升不同能力级别模型的研究方向截然不同。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决现有视觉代理评测过于简单、仅局限于单轮对话或静态图片、缺乏状态化工具交互的问题。
*   **现有痛点**：现有框架多为纯文本导向，即便有视觉输入，也多为静态前缀，无法模拟多轮交互中图片动态进入的复杂性，且工具空间过小，无法反映真实助手需求。
*   **研究假设**：视觉工具调用是一项独立的代理能力，需要将感知与行动深度解耦，且随着模型规模增长，其核心短板会从“任务规划”转向“视觉感知精度”。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **场景生成流水线**：先进行图像-工具关联（筛选 actionable 图像），通过 CLIP 聚类分组，利用 LLM 进行信息流规划，最后生成包含目标状态与预期结果的脚本。
    2.  **执行模式**：提供“代码执行”模式（模型输出 Python 代码）和“结构化工具调用”模式（Schema 限制）。
    3.  **状态评估**：通过对比初始与最终世界状态的实体差异（Entity F1）进行确定性评估。
    4.  **智能体判断**：使用 Claude 4.5 Opus 作为 Agent Judge 进行 rubric 评分，涵盖任务完成度、指令遵循、工具调用有效性等。
*   **关键模块**：
    *   **Tool-Discovery**：通过元工具 `search_tool` 实现按需检索，解决 500+ 工具无法全部放入 Prompt 的上下文瓶颈。
    *   **UI 模式**：通过自定义 UI 工具实现代理对交互界面的渲染，解决单纯文本交互的限制，验证代理构建功能性界面的能力。

---

### 4. 方法对比分析
*   **本质区别**：从“静态图片+单轮任务”向“多图像+动态状态化交互”演进，强调感知与行动的闭环。
*   **创新贡献**：提出了“规划-精度（Planning-to-Precision）”故障模型交叉理论，明确了模型规模与核心瓶颈间的非线性关系。
*   **适用场景**：适用于需要复杂工具调用、多模态信息融合、以及长期状态记忆的各类 AI 代理开发与评测。

---

### 5. 实验分析
*   **验证方法**：在 12 个强力模型上进行横向对比，并分析了模型规模、信息流、图像到达模式对成功率的影响。
*   **关键结论**：最强模型 Claude 4.5 Opus 成功率不到 50%。随着参数增加，模型从“不会规划”转变为“看不清细节”。
*   **优势**：评估维度全面，具备高度的可复现性和明确的瓶颈定位能力。
*   **局限**：对 LLM Judge 的依赖可能导致一定的评估偏见，且自动化生成流水线本身可能存在模型生成偏好。

---

### 6. 实用指南
*   **开源地址**：[https://github.com/apple/ml-mmtoolsandbox](https://github.com/apple/ml-mmtoolsandbox)
*   **实现建议**：复现时重点关注 `Oracle` 合规性检查，利用预定义的 511 个工具集测试模型在处理大规模工具注册表时的调用检索能力。
*   **迁移可能**：可直接迁移至桌面级或工业级自动化代理的评估，特别是在需要处理复杂 GUI 的任务中。

---

### 7. 总结
*   **核心思想**：视觉工具调用能力的瓶颈随模型规模进化，从规划逻辑转向高精度感知。
*   **速记版 Pipeline**：
    1.  **准备环境**：加载状态化的工具库及模拟用户。
    2.  **感知输入**：代理处理图片并触发检索工具。
    3.  **规划与执行**：代理输出代码或调用 API 执行任务。
    4.  **环境更新**：评估执行后的实体状态变化。
    5.  **反馈循环**：LLM 裁判对结果打分，记录故障类型。

**Key Findings:**

- We introduce MM-ToolSandBox, a benchmark and evaluation framework for visually grounded tool-calling agents.
- Evaluating 12 state-of-the-art models, from 4B open-weight to frontier proprietary systems, shows that current models still lack robust visual tool-calling capability: even the best model achieves below 50% success rate.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11818v1)
- [arXiv](https://arxiv.org/abs/2607.11818v1)

---

<a id='2607.11739v1'></a>
## [AutoPath: Learning Transferable Goal-Conditioned Stochastic Path Prior for Safe Navigation Without Human Demonstrations](https://arxiv.org/abs/2607.11739v1)

**Authors:** Ziyang Zhang, Boyang Zhou, Zesong Yang, Haocheng Peng, Zeming Gai, Xiao Liang, Yujun Shen, Danping Zou, Ruizhen Hu, Hujun Bao, Zhaopeng Cui

**Published:** 2026-07-13

**Categories:** cs.RO

**Abstract:**

Real-time navigation in cluttered and dynamic environments requires collision-free and dynamically feasible motion under limited perception. However, feasible navigation behaviors are inherently multimodal because multiple paths may exist around obstacles. In this paper, we formulate navigation as learning a transferable goal-conditioned stochastic path prior that models a reusable distribution over goal-aligned geometry-consistent local paths conditioned on local observations. This formulation enables structured sampling of navigation candidates, allowing multiple feasible paths to be explored through sampling without relying on robot-specific motion constraints. To this end, we introduce a goal-aligned canonical state representation that removes in-plane rotational ambiguity and normalizes local geometry with respect to the goal, enabling rotation-invariant path distribution learning. We further develop a structured prior learning framework that parameterizes local paths using a geometry-aware polar action manifold and incorporates risk-sensitive utility shaping with multi-goal distributional rollouts for stable and safety-aware planning. Extensive experiments in dense static environments and dynamic pedestrian scenarios demonstrate that the proposed method achieves consistently high success rates with competitive efficiency while enabling cross-platform transfer of a single path prior learned on differential-drive robots to quadruped platforms without retraining.

**Analysis:**

# AutoPath 论文分析

## 1. 摘要翻译
实时导航在复杂动态环境中需要满足避障与动力学可行性。由于绕过障碍物存在多种路径选择，导航行为本质上是多模态的。本文提出 **AutoPath**，一种可迁移的“目标条件随机路径先验”学习方法。它建模了目标对齐的局部路径分布，消除了特定机器人的动力学约束。通过引入“目标对齐的规范状态表示（goal-aligned canonical state）”，消除了平面内的旋转模糊并对局部几何进行了归一化，实现了路径分布学习的旋转不变性。结合几何感知极坐标动作流形与风险敏感的效用塑造，该方法实现了稳定的先验学习，并支持无人类演示的跨平台迁移（如从差分驱动机器人到四足机器人）。

## 2. 方法动机分析
- **驱动力**：打破“导航策略与机器人具体动力学强耦合”的现状，通过解耦实现导航能力的通用性与可迁移性。
- **痛点**：现有学习型导航方法通常将多模态路径坍缩为单一确定性动作，或将空间推理与底层的动力学约束强绑定，导致无法跨平台复用，且泛化能力弱。
- **研究假设**：通过在“目标对齐的规范空间”中学习通用的路径分布，可以将复杂的动力学决策问题简化为几何推理问题，从而实现底层的通用性和上层的适应性。

## 3. 方法设计详解
- **核心 Pipeline**：
  1. **状态规范化**：将 LiDAR 数据和目标点转换到以机器人为原点、x轴指向目标点的规范坐标系 $F_s$。这使得导航任务对环境旋转具有不变性。
  2. **先验模型学习**：在 $F_s$ 中，通过神经网络预测局部路径的分布。采用几何感知极坐标动作流形（Polar Action Manifold），将路径表示为一组径向排序的极坐标控制点，降低了搜索维度。
  3. **多目标采样与推演**：在 inference 阶段，根据当前目标半径进行圆周采样产生多个备选目标，对每个目标采样多种路径。
  4. **约束优化精炼**：利用 Trajectory Optimization (如 Priest 优化器) 将采样路径转化为满足具体平台动力学约束（如最大速度、曲率限制）的最终轨迹。
- **关键公式与逻辑**：
  - **规范化 (Eq 4)**：$x^{(s)} = R(-\psi_t)x^{(r)}$，通过旋转抵消了方位角的影响，使模型只关注“目标在哪里”而非“自身朝向哪里”。
  - **风险敏感 utility (Eq 10)**：将 success, safety, smoothness 等项归一化并加权，通过 CVaR 机制评估路径风险，确保生成轨迹不仅“能走”而且“安全”。

## 4. 方法对比分析
- **本质区别**：从“预测直接控制指令”转向“预测空间路径概率分布”。它不直接输出电机控制量，而是输出几何路径，具体的控制由后续的优化模块处理。
- **创新贡献**：提出了一种与物理实体解耦的规范化路径表达方式，通过离线合成数据训练通用先验，通过在线采样和约束优化实现跨平台迁移。
- **适用场景**：复杂动态障碍物环境、多种异构机器人平台下的自主导航。

## 5. 实验分析（精简版）
- **验证方法**：在 Gazebo 和 Isaac Sim 模拟器中对比了 DRL-VO, PathRL, CrowdSurfer 等主流方法，并在 TurtleBot4 和 Unitree Go2 上进行实地部署。
- **关键结果**：在复杂动态环境下，AutoPath 的成功率显著高于基线（如提高 8%-15%）；实验证明其在完全未见过的环境下具有极强的零样本迁移能力。
- **主要优势**：极强的泛化性与可迁移性，无需针对不同机器人重新训练模型。
- **主要局限**：目前的方案未显式建模多智能体间的交互协作，仅将其视为动态障碍物处理。

## 6. 实用指南
- **迁移建议**：若要迁移至新平台，仅需修改代码中的约束集合 $K$（如动力学参数限制）和嵌入式优化器的输入参数，无需重新训练先验模型。
- **实现细节**：
  - 务必保证规范化坐标系 $F_s$ 的构建准确，这是模型泛化的灵魂。
  - 训练时 Stage I 离线初始化至关重要，它提供了高质量的路径先验。
  - 推理时若 CPU 计算紧张，可适当减少圆周采样点数 $K$。

## 7. 总结
- **核心思想**：学习通用的几何路径分布，将导航转化为跨平台的几何推理问题。
- **速记版 Pipeline**：
  1. **坐标对齐**：将环境旋转至以目标为准的规范坐标系。
  2. **路径采样**：基于先验概率分布生成多模态路径候选。
  3. **评估筛选**：根据安全与几何特征筛选最优候选。
  4. **动力学优化**：根据机器人的具体约束微调路径执行。

**Key Findings:**

- To this end, we introduce a goal-aligned canonical state representation that removes in-plane rotational ambiguity and normalizes local geometry with respect to the goal, enabling rotation-invariant path distribution learning.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11739v1)
- [arXiv](https://arxiv.org/abs/2607.11739v1)

---

<a id='2607.11734v1'></a>
## [NeuralActuator: Neural Actuation Modeling for Robot Dynamics and External Force Perception](https://arxiv.org/abs/2607.11734v1)

**Authors:** Zhiyang Dou, John U. Onyemelukwe, Hangxing Zhang, Heng Zhang, Minghao Guo, Yunsheng Tian, Michal Piotr Lipiec, Joshua Jacob, Chao Liu, Peter Yichen Chen, Yuri Ivanov, Wojciech Matusik

**Published:** 2026-07-13

**Categories:** cs.RO, cs.CV, cs.GR, cs.LG

**Abstract:**

Differentiable simulators have advanced policy learning and model-based control, yet actuator dynamics remain an important source of sim-to-real error. This is particularly acute on low-cost platforms, where the linear current-to-torque relation $τ= K_tI$ becomes unreliable during commanded-target tracking because of friction, hysteresis, backlash, and thermal effects. We present NeuralActuator, a neural actuator model that jointly predicts (i) a simulator-equivalent generalized-effort surrogate for trajectory propagation on low-cost servo platforms, (ii) external force with a contact-probability gate for sensorless force perception, and (iii) a motor-condition score for the supervised joint. We also introduce the Neural Actuation Dataset (NAD), collected with a twin-arm teleoperation system that records robot states and actuator telemetry together with external-force labels. The torque-surrogate head is trained through differentiable simulation from pose trajectories without direct generalized-effort labels, while the force, gate, and motor-condition heads receive direct supervision. A Transformer captures temporal dependencies while supporting real-time inference. We evaluate NeuralActuator on a 5-DoF OpenManipulator-X, a 6-DoF SO-101, and a 7-DoF Franka Emika Panda, spanning three actuator families and platforms costing approximately USD 500 to over USD 30,000. The low-cost platforms support dynamics and force evaluation, while the offline Franka experiment provides an additional payload-force-estimation benchmark. Experiments further demonstrate its application for motor condition estimation on OpenManipulator-X and improved behavior-cloning performance when NeuralActuator is used as a pretrained module.

**Analysis:**

这是一篇极具前瞻性的机器人学与机器学习交叉领域论文。以下是对 **NeuralActuator** 的专业分析：

### 1. 论文核心贡献总结
NeuralActuator 提出了一种通用的神经驱动建模框架，旨在解决低成本机器人平台中“驱动器动力学复杂（如摩擦、迟滞、热效应）”导致的仿真与现实差距（Sim-to-Real gap）。该框架通过一个统一的 Transformer 模型实现了精确的驱动力矩代理预测、无传感器接触力估计以及电机健康状态监控，显著提升了低成本机器人在动力学仿真与行为克隆任务中的表现。

### 2. 核心创新点与方法论
*   **多任务驱动建模**：通过一个多头 Transformer 架构，同时处理三个关键任务：力矩代理预测（替代复杂的物理方程）、接触概率门控（用于无传感器力感知）以及电机条件评分。
*   **无标签自监督学习**：其“力矩代理头（Torque-surrogate head）”的训练无需显式的力矩标签，而是通过**可微仿真（Differentiable Simulation）**直接从位姿轨迹中学习，这种端到端的闭环优化方式极具创新性。
*   **Neural Actuation Dataset (NAD)**：构建了首个跨硬件平台的驱动器遥操作数据集，填补了低成本伺服电机在真实环境下的动力学数据空白。

### 3. 对计算机视觉及机器人领域的潜在影响
*   **Sim-to-Real 的范式转移**：传统 Sim-to-Real 往往依赖于域随机化（Domain Randomization），而 NeuralActuator 提供了一种“数据驱动的动力学矫正”范式，能够让低廉的硬件表现出接近工业级的动力学一致性。
*   **视觉与控制的融合（计算机视觉视角）**：对于 CV 研究者而言，这意味着在进行机器人操纵任务（Manipulation）时，可以通过 NeuralActuator 获得高精度的动力学先验，从而减少视觉感知与物理执行之间的断层。如果将该模型集成到视觉模仿学习（Imitation Learning）中，将大幅提升视觉策略在现实世界的鲁棒性。

### 4. 潜在的应用领域
*   **低成本机器人开发**：使廉价的开源机械臂（如 OpenManipulator）能够执行需要精确力控的任务（如精细装配）。
*   **基于触觉的无传感器感知**：在没有力矩传感器的情况下，仅通过电机电流与位姿 telemetry 实现“伪触觉”，降低机器人硬件成本。
*   **预测性维护（Predictive Maintenance）**：通过 motor-condition head 实现对工业机器人及服务机器人的实时磨损监测。

### 5. 可推断的局限性
*   **跨平台泛化能力的边界**：尽管测试了从 500 美元到 30,000 美元的平台，但该模型是否能在极其复杂的动态环境下（如高频振动、极端负载变化）保持性能仍待验证。
*   **可微仿真的计算开销**：虽然推理阶段由 Transformer 完成（支持实时），但训练阶段依赖可微仿真，对于更高自由度（DoF）或更复杂环境的扩展，仿真器的构建和优化难度可能会成为瓶颈。
*   **对传感器数据的依赖**：该模型依赖于驱动器遥测数据（Telemetry），如果底层控制器的反馈频率或精度受限，模型的预测上限会受到硬件接口的物理约束。

**专家点评：**
这篇论文的亮点在于它并没有试图去“简化”物理世界，而是通过学习动力学的“剩余误差”来增强仿真。对于计算机视觉研究者来说，NeuralActuator 提供了一个优秀的**物理先验模块**。当你在处理视觉引导的抓取或操作任务时，利用该模块可以有效滤除动作执行中的非线性噪声，从而让模型更专注于视觉特征的学习，而非被迫学习复杂的动力学扰动。

**Key Findings:**

- We present NeuralActuator, a neural actuator model that jointly predicts (i) a simulator-equivalent generalized-effort surrogate for trajectory propagation on low-cost servo platforms, (ii) external force with a contact-probability gate for sensorless force perception, and (iii) a motor-condition score for the supervised joint.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11734v1)
- [arXiv](https://arxiv.org/abs/2607.11734v1)

---

<a id='2607.11638v1'></a>
## [DA-Nav: Direction-Aware City-Scale Vision-Language Navigation](https://arxiv.org/abs/2607.11638v1)

**Authors:** Ye Yuan, Kehan Chen, Xinqiang Yu, Wentao Xu, Heng Wang, Libo Huang, Chuanguang Yang, Yan Huang, Jiawei He, Zhulin An

**Published:** 2026-07-13

**Categories:** cs.RO

**Abstract:**

City-scale outdoor navigation is currently hindered by the heavy reliance on dense maps or costly navigation supervision. In this work, we introduce a novel paradigm for leveraging directional instructions from commercial navigation tools (e.g., Google Maps). To bridge the gap between commercial instructions and executable navigation actions, while mitigating long-horizon error accumulation through robust trajectory recovery, we propose DA-Nav, a Direction-Aware vision-language Navigation framework that reformulates navigation as a discrete spatial grounding problem on the egocentric 2D image plane. To achieve trajectory recovery, DA-Nav employs a Chain-of-Thought (CoT) reasoning process encompassing deviation assessment, action prediction, and target grid selection. We further introduce ReDA, a dataset that provides direction-aware instructions and recovery trajectories to enhance spatial grounding and support CoT recovery reasoning. Extensive experiments in CARLA demonstrate that DA-Nav achieves a high success rate of 56.16% in unseen urban environments, outperforming existing State-of-The-Art (SoTA) methods while maintaining a substantially stronger recovery capability. Furthermore, without fine-tuning, DA-Nav seamlessly adapts to both quadruped and humanoid robots, enabling stable kilometer-scale closed-loop outdoor navigation in complex real world environments.

**Analysis:**

### 1. 摘要翻译
城市级户外导航目前受限于对高精地图或昂贵人工导航监督的过度依赖。本文引入了一种新范式，利用商业导航工具（如高德地图）提供的方向性指令进行导航。为了弥合商业指令与可执行导航动作之间的差距，并缓解长距离导航中的误差累积，我们提出了DA-Nav，一个基于方向感知的视觉语言导航框架，将导航重新建模为以视觉语言为条件的二维图像平面上的离散空间接地问题。为实现轨迹恢复，DA-Nav采用思维链（CoT）推理过程，涵盖偏差评估、动作预测和目标网格选择。此外，我们引入了ReDA数据集，提供方向感知指令和恢复轨迹，以增强空间接地能力并支持CoT恢复推理。在CARLA中的实验表明，DA-Nav在未见过的城市环境中达到了56.16%的成功率，超越了现有的SoTA方法，并具备强大的恢复能力。此外，无需微调，DA-Nav即可无缝适配四足和人形机器人，实现复杂现实环境中的公里级闭环导航。

### 2. 方法动机分析
*   **驱动力**：利用现成、易获取的商业导航工具作为“大脑”，替代昂贵的高精地图和精细化标注，实现低成本的机器人城市级自主导航。
*   **现有方法痛点**：传统方法依赖SLAM导致维护成本高，且对动态环境敏感；现有VLN方法依赖极度精细的人工标注，难以扩展到城市规模；纯连续轨迹回归易产生累积误差，且缺乏针对偏离路径的有效恢复机制。
*   **研究假设**：通过将“导航指令”转化为“图像平面上的离散空间目标”，并引入基于思维链（CoT）的自我偏差纠正机制，可以显著提高长周期导航的鲁棒性和泛化能力。

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **输入处理**：输入连续 egocentric 观察 $O_t$ 和商业导航离散指令 $I_t$（如“左转”）。
    2.  **思维链（CoT）推理**：模型以自回归方式依次生成三个决策：
        *   **偏差评估** ($s_t$)：判断当前是否偏离航线（Yes/No）。
        *   **动作预测** ($c_t$)：输出语义动作（如 FORWARD, CORRECT_LEFT 等）。
        *   **目标预测** ($P_t$)：在图像平面网格 $G$ 上预测轨迹坐标序列。
    3.  **空间投影**：将网格坐标通过摄像机内参及深度信息（或平面假设）投影为机器人本体坐标系下的3D路径。
    4.  **控制执行**：采用“最远点跟踪”策略，选择路径上的最远点作为控制目标，配合heading-error驱动的角速度控制器。
*   **模型结构**：基于 Qwen2.5-VL-7B 骨干网，利用 LoRA 技术进行参数高效微调，冻结视觉编码器和LLM主干，仅训练适应导航任务的适配层。

### 4. 方法对比分析
*   **本质区别**：DA-Nav 从“回归连续路径坐标”转变为“图像平面上的离散空间选择”，将导航从运动控制问题转化为视觉语言模型具备的长项——空间接地推理问题。
*   **创新贡献**：引入了显式的偏差恢复机制（CoT-based recovery）和对应的ReDA数据集，不仅训练了导航，还训练了机器人“知错能改”的能力。
*   **适用场景**：适用于各类具有视觉观察能力的移动平台（轮式、四足、人形），特别是在GPS精度不佳或地图未覆盖的复杂动态城市环境。

### 5. 实验分析（精简版）
*   **关键结果**：在模拟器中达到59%的成功率（SR），且在现实世界中实现了1.2公里的零样本长距离导航。
*   **主要优势**：极强的自我修正能力（CSR高达98%），克服了传统行为克隆方法在遇到偏离时“一错到底”的局限。
*   **主要局限**：推理依赖远程GPU算力，且在商业导航工具无法覆盖的GPS盲区或极度非结构化场景中仍存在依赖限制。

### 6. 实用指南
*   **实现细节**：
    *   **数据构建**：使用FSM在模拟器中模拟偏离状态（DRIFTING），随后进入恢复状态（RECOVERING），这对训练鲁棒性至关重要。
    *   **训练建议**：必须冻结预训练大模型底座，仅通过LoRA微调，否则会破坏预训练模型的通用理解能力。
*   **迁移建议**：该框架核心是“指令-图像-决策”的映射。迁移至其他任务只需更换指令集和对应的动作空间定义，无需更改主干模型。

### 7. 总结
*   **核心思想**：利用思维链推理与图像平面接地，实现具身智能的自我导航与偏差恢复。
*   **速记版 Pipeline**：
    1. 输入视觉观察与导航指令。
    2. 大模型自回归输出“是否偏离、采取动作、轨迹坐标”序列。
    3. 将轨迹点映射回机器人物理坐标系。
    4. 执行最远点跟踪控制以实现移动。

**Key Findings:**

- In this work, we introduce a novel paradigm for leveraging directional instructions from commercial navigation tools (e.g., Google Maps).
- To bridge the gap between commercial instructions and executable navigation actions, while mitigating long-horizon error accumulation through robust trajectory recovery, we propose DA-Nav, a Direction-Aware vision-language Navigation framework that reformulates navigation as a discrete spatial grounding problem on the egocentric 2D image plane.
- Extensive experiments in CARLA demonstrate that DA-Nav achieves a high success rate of 56.16% in unseen urban environments, outperforming existing State-of-The-Art (SoTA) methods while maintaining a substantially stronger recovery capability.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11638v1)
- [arXiv](https://arxiv.org/abs/2607.11638v1)

---

<a id='2607.11588v1'></a>
## [FoundationGeo: Learning Spatial Pixel-Wise Fields for Monocular Metric Geometry](https://arxiv.org/abs/2607.11588v1)

**Authors:** Muxin Liu, Xiaoyang Lyu, Tianhe Ren, Peng Dai, Xiaoshan Wu, Zhiyue Zhang, Jiaqi Zhang, Jiehong Lin, Shaoshuai Shi, Xiaojuan Qi

**Published:** 2026-07-13

**Categories:** cs.CV

**Abstract:**

We present FoundationGeo, a two-stage framework that explicitly bridges relative and metric prediction via spatial calibration and principled data design. Stage 1 learns a high-fidelity, affine-invariant geometry model by initializing with DINOv3 and training on a curated 10.2M-sample multi-domain corpus with complementary local-detail supervision, yielding sharp boundaries and strong cross-domain generalization. Stage 2 moves beyond global scaling by introducing lightweight pixel-wise calibration fields for metric estimation: a scale field for spatially varying metric alignment and a ray-direction correction field that mitigates directional bias in point-map geometry, together producing metrically consistent 3D point maps. Beyond model design, we identify camera intrinsic coverage, especially focal length distribution mismatch between training and test data, as a key bottleneck for zero-shot metric generalization: performance drops sharply when test intrinsics fall outside the training distribution. To address this, we synthesize additional training data across diverse focal lengths using a Blender-based data engine, repairing under-covered focal regimes and improving robustness under intrinsic shift. Extensive zero-shot evaluations across seven benchmarks show that FoundationGeo significantly strengthens cross-domain robustness, staying near the top across diverse domains while avoiding the sharp cross-domain performance drops observed in other methods. This consistency translates into the best overall performance, surpassing heavier baselines by over 5.2% on average.

**Analysis:**

### 1. 摘要翻译
我们提出了 FoundationGeo，这是一个通过空间校准和原则性数据设计，显式连接相对深度与度量深度预测的双阶段框架。第一阶段通过 DINOv3 初始化并训练 10.2M 样本的多领域语料库，结合局部细节监督，学习高保真、仿射不变的几何模型，从而实现锐利的边界和强大的跨领域泛化。第二阶段通过引入轻量级像素级校准场，超越了全局缩放的限制：利用空间变化的尺度场进行度量对齐，以及射线方向校正场来减轻点图几何中的方向偏差，共同生成度量一致的 3D 点云。此外，我们将相机内参覆盖率（特别是训练与测试数据间的焦距分布不匹配）确认为零样本度量泛化的关键瓶颈，并构建了一个基于 Blender 的数据引擎合成多焦距训练数据，显著提升了跨域鲁棒性。

---

### 2. 方法动机分析
*   **驱动力**：旨在填补“相对几何预测（高精度/泛化强）”与“度量深度估计（尺度感知/物理一致）”之间的鸿沟，解决现有单目深度估计在跨领域和相机模型切换时的性能退化问题。
*   **现有方法痛点**：
    1.  现有方法多依赖单一全局尺度进行度量对齐，无法捕捉空间变化的尺度偏差。
    2.  忽视了“射线方向偏差”：即使尺度对齐，错误的射线方向仍会导致点云几何形状畸变。
    3.  训练数据的焦距分布单一，导致模型对未见过的内参（相机模型）表现脆弱。
*   **研究假设**：度量估计的瓶颈不仅在于相对基础模型的强弱，更在于空间畸变（尺度与方向）的建模和训练数据中内参的多样性。

---

### 3. 方法设计详解
*   **阶段一：升级相对基础模型**
    *   使用 DINOv3-ViT 提取特征，CNN 解码器回归仿射不变点图 $\hat{\mathbf{P}}$ 和可靠性掩码 $\hat{\mathbf{M}}$。
    *   引入多尺度局部损失和边缘损失，以确保高频几何细节。
*   **阶段二：空间校准场（核心创新）**
    *   **射线方向校正 ($\Delta$)**：为每个像素构建局部正交切平面（基于参考轴与当前射线方向的叉乘），预测切平面内的 2D 偏移，对射线方向进行微调，且保持原始距离（range）不变，从而解耦方向偏差与尺度修正。
    *   **像素级尺度场 ($S$)**：预测每个像素的缩放系数，直接通过 $\tilde{\mathbf{P}} = S \odot \hat{\mathbf{P}}'$ 实现空间自适应对齐。
    *   **训练目标**：采用耦合度量 $l_1$ 损失进行总体约束，辅以针对 $\Delta$ 和 $S$ 的解耦损失，确保校准模块不冗余吸收几何信息。

---

### 4. 方法对比分析
*   **本质区别**：从传统的“全局缩放”转变为“像素级空间场校准”；从“被动数据混合”转变为“基于焦距分布瓶颈的主动数据合成”。
*   **创新点**：提出了射线方向校正场，解决了传统方法中“对齐了尺度但点云结构仍歪曲”的问题。
*   **适用场景**：极度依赖度量准确性的场景（如 AR/VR、机器人交互、室外/室内混合的开放领域任务）。

---

### 5. 实验分析
*   **结果**：在七个不同基准测试中，AbsRel 平均提升 5.7%，$\delta_1$ 提升 5.2%。在 HAMMER 等具有挑战性的对象中心数据集上表现出极强的鲁棒性。
*   **主要优势**：极强的跨域泛化能力，对不同相机型号表现稳定。
*   **局限**：校准场目前是轻量级的，处理极端出分布（OOD）场景的能力仍受限于基础模型的相对几何感知上限。

---

### 6. 实用指南
*   **开源**：[https://mx-liu6.github.io/FoundationGeo-web/](https://mx-liu6.github.io/FoundationGeo-web/)
*   **实现细节**：
    *   训练数据滤波至关重要：去除鸟瞰图、曝光严重受损和无近-远顺序的图像。
    *   参数配置：$\gamma_s, \gamma_r, \gamma_\Delta$ 分别设为 0.2, 0.1, 0.05。
    *   合成数据：通过 Blender 渲染不同焦距图像，手动设计轨迹确保多样性，而非随机采样。
*   **迁移**：该校准模块（$\Delta$ 和 $S$ 场）可独立作为一个轻量级“后处理/微调插件”，迁移到任何现有的相对深度预测模型之上。

---

### 7. 总结
*   **核心思想**：通过空间场校准解决局部几何畸变，通过焦距多样化提升泛化性。
*   **速记版pipeline**：
    1.  训练强力相对深度基础模型。
    2.  预测相对深度和切平面校正偏移。
    3.  通过校正场修正射线方向。
    4.  通过像素级缩放场对齐物理尺度。

**Key Findings:**

- We present FoundationGeo, a two-stage framework that explicitly bridges relative and metric prediction via spatial calibration and principled data design.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.11588v1)
- [arXiv](https://arxiv.org/abs/2607.11588v1)

---

