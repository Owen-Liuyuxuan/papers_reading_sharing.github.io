time: 20260319

# Arxiv Computer Vision Papers - 2026-03-19

## Executive Summary

### **Arxiv 计算机视觉领域论文日报执行摘要 (2026-03-17)**

**1. 核心主题与趋势**

今日的论文集合清晰地反映了计算机视觉领域的几个强劲趋势：

*   **三维与具身智能的深度融合**：超过一半的论文（如 MolmoB0T, Loc3R-VLM, LoST, GMT, Feeling the Space, AHOY!）聚焦于3D场景理解、物体操作和具身智能。研究重点从“看”3D转向“用”3D进行推理、规划和交互，标志着向更实用、更物理化的AI系统演进。
*   **视频理解的效率与长上下文挑战**：多篇论文（VideoAtlas, Unified Spatio-Temporal Token Scoring）致力于解决长视频处理的计算瓶颈，通过创新的token选择、压缩或索引技术，实现对海量视频数据的高效分析与导航。
*   **多模态大模型的垂直化与能力拓展**：研究不再局限于通用的视觉-语言模型（VLM），而是为其注入**特定领域能力**，如基于语言的3D定位（Loc3R-VLM）、6自由度轨迹合成（GMT）和无需训练的精细化视频编辑（Versatile Editing），展示了VLM专业化应用的巨大潜力。
*   **数据与先验的创造性利用**：出现了利用大规模仿真数据实现零样本操作（MolmoB0T）、结合高斯溅射与视频扩散先验重建复杂人体（AHOY!），以及自适应雷达压缩（AdaRadar）等工作，表明在模型架构创新之外，对**数据生成、信号处理和先验知识融合**的探索同样关键。

**2. 突出创新与重要论文**

*   **MolmoB0T (Deshpande et al.)**：**最具突破性潜力**。提出通过“大规模仿真”实现“零样本操作”，若其方法成立，可能为机器人学习提供一种全新的、低成本、高可扩展性的范式，绕过昂贵且缓慢的真实世界数据收集。
*   **VideoAtlas (Eltahir et al.)**：**极具实用价值**。针对长视频分析这一痛点，提出“对数级计算”的导航方法。若能实现其宣称的效率提升，将对视频内容检索、摘要和审核等工业应用产生直接影响。
*   **AHOY! (Mir et al.)**：**技术整合的典范**。巧妙地将**高斯溅射（3D重建）、视频扩散先验（生成先验）与遮挡处理**结合，从单目视频中创建可动画化、高保真的人体模型，代表了3D数字人技术的前沿方向。
*   **Versatile Editing (Kulikov et al.)**：**简洁而强大**。强调“无需训练”即可对视频内容、动作和动态进行多样化编辑，降低了高质量视频编辑的门槛，是即插即用型工具的重要进展。

**3. 新兴研究方向**

*   **仿真到现实的零样本迁移**：MolmoB0T 预示着一个新方向，即构建高度物理真实、任务多样化的仿真环境，作为训练通用具身智能体的主要数据源。
*   **VLM 的 3D 空间推理与具身化**：Loc3R-VLM 和 GMT 展示了将VLM的语义理解与具体的3D坐标、轨迹和动作生成相结合的趋势，使VLM成为机器人或虚拟代理的“空间大脑”。
*   **感知的模态自适应与高效压缩**：AdaRadar 提出的雷达信号自适应压缩，以及多篇论文对视频token的高效处理，表明在边缘计算和资源受限场景下，“**自适应感知**”和“**计算-精度-带宽**”的智能权衡成为关键课题。

**4. 推荐精读论文**

根据研究者的兴趣方向，建议优先阅读：

*   **所有研究者**：**VideoAtlas**。其高效长视频处理思路具有普适的启发意义。
*   **机器人/具身AI方向**：**MolmoB0T**（必读）和 **GMT**。前者关乎范式，后者关乎具体技术。
*   **3D视觉/图形学方向**：**AHOY!** 和 **LoST**。前者是前沿应用，后者是3D形状表征的基础创新。
*   **视频理解/多模态方向**：**Unified Spatio-Temporal Token Scoring** 和 **Versatile Editing**。前者关乎效率核心，后者关乎应用创新。
*   **高效/边缘计算方向**：**AdaRadar** 和 **Feeling the Space**。关注特定传感器和场景下的优化设计。

---
**总结**：本日论文展现了计算机视觉从静态图像感知向动态、三维、可交互且高效的智能系统快速演进的图景。研究前沿集中在**利用仿真与先验数据、赋予VLM空间行动能力，以及攻克长序列处理效率**三大阵地上。建议根据上述分类，选取相关论文深入研读。

---

## Table of Contents

1. [MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation](#2603.16861v1)
2. [Unified Spatio-Temporal Token Scoring for Efficient Video VLMs](#2603.18004v1)
3. [Loc3R-VLM: Language-based Localization and 3D Reasoning with Vision-Language Models](#2603.18002v1)
4. [LoST: Level of Semantics Tokenization for 3D Shapes](#2603.17995v1)
5. [GMT: Goal-Conditioned Multimodal Transformer for 6-DOF Object Trajectory Synthesis in 3D Scenes](#2603.17993v1)
6. [Versatile Editing of Video Content, Actions, and Dynamics without Training](#2603.17989v1)
7. [Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding](#2603.17980v1)
8. [AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception](#2603.17979v1)
9. [AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors](#2603.17975v1)
10. [VideoAtlas: Navigating Long-Form Video in Logarithmic Compute](#2603.17948v1)

---

## Papers

<a id='2603.16861v1'></a>
## [MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation](https://arxiv.org/abs/2603.16861v1)

**Authors:** Abhay Deshpande, Maya Guru, Rose Hendrix, Snehal Jauhri, Ainaz Eftekhar, Rohun Tripathi, Max Argus, Jordi Salvador, Haoquan Fang, Matthew Wallingford, Wilbert Pumacay, Yejin Kim, Quinn Pfeifer, Ying-Chun Lee, Piper Wolters, Omar Rayyan, Mingtong Zhang, Jiafei Duan, Karen Farley, Winson Han, Eli Vanderbilt, Dieter Fox, Ali Farhadi, Georgia Chalvatzaki, Dhruv Shah, Ranjay Krishna

**Published:** 2026-03-17

**Categories:** cs.RO

**Abstract:**

A prevailing view in robot learning is that simulation alone is not enough; effective sim-to-real transfer is widely believed to require at least some real-world data collection or task-specific fine-tuning to bridge the gap between simulated and physical environments. We challenge that assumption. With sufficiently large-scale and diverse simulated synthetic training data, we show that zero-shot transfer to the real world is not only possible, but effective for both static and mobile manipulation. We introduce MolmoBot-Engine, a fully open-source pipeline for procedural data generation across robots, tasks, and diverse simulated environments in MolmoSpaces. With it, we release MolmoBot-Data, a dataset of 1.8 million expert trajectories for articulated object manipulation and pick-and-place tasks. We train three policy classes: MolmoBot, a Molmo2-based multi-frame vision-language model with a flow-matching action head; MolmoBot-Pi0, which replicates the $π_0$ architecture to enable direct comparison; and MolmoBot-SPOC, a lightweight policy suitable for edge deployment and amenable to RL fine-tuning. We evaluate on two robotic platforms: the Franka FR3 for tabletop manipulation tasks and the Rainbow Robotics RB-Y1 mobile manipulator for door opening, drawer manipulation, cabinet interaction, and mobile pick-and-place. Without any real-world fine-tuning, our policies achieve zero-shot transfer to unseen objects and environments. On tabletop pick-and-place, MolmoBot achieves a success rate of 79.2% in real world evaluations across 4 settings, outperforming $π_{0.5}$ at 39.2%. Our results demonstrate that procedural environment generation combined with diverse articulated assets can produce robust manipulation policies that generalize broadly to the real world. Technical Blog: https://allenai.org/blog/molmobot-robot-manipulation

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **MolmoB0T** 的论文进行了深入分析。以下是详细评估：

### 1. 核心贡献总结
MolmoB0T 挑战了“机器人学习必须依赖真实世界数据进行微调”的传统认知，证明了通过大规模、多样化的合成模拟数据，可以实现卓越的零样本（Zero-Shot）仿真到现实（Sim-to-Real）迁移。研究团队推出了完全开源的 **MolmoBot-Engine** 和包含 180 万条专家轨迹的 **MolmoBot-Data** 数据集，在不进行任何现实世界微调的前提下，实现了对复杂机械臂和移动操作任务的稳健控制。

### 2. 关键创新与方法论
*   **模拟数据规模与多样性的质变**：该研究利用 **MolmoSpaces** 进行程序化环境生成，不仅仅是增加数据量，而是通过多样化的资产（articulated assets）覆盖了复杂的物体交互，从本质上降低了“仿真-现实”间隙。
*   **多模型范式验证**：论文并未止步于单一架构，而是对比了三种策略：基于 VLM 的 **MolmoBot**（利用 Molmo2 视觉语言模型能力）、复刻 $\pi_0$ 架构的 **MolmoBot-Pi0**，以及轻量级的 **MolmoBot-SPOC**。这种对比展示了视觉-语言模型（VLM）作为通用策略底座的巨大潜力。
*   **流匹配（Flow-Matching）动作头**：通过采用流匹配技术，模型能够更平滑、精准地预测操作轨迹，克服了传统模仿学习中动作预测的抖动问题。

### 3. 对领域的潜在影响
*   **范式转移（Paradigm Shift）**：如果“大规模模拟”确实能完全替代昂贵的现实数据收集，这将极大地降低机器人开发的门槛，使通用机器人从实验室走向现实世界的步伐显著加快。
*   **VLM 作为机器人大脑的标准化**：论文强力支撑了“视觉-语言模型（VLM）是通用的机器人基础模型”这一观点。MolmoBot-Engine 的开源可能使其成为未来研究“具身智能”的一个基准框架，类似于计算机视觉领域的 ImageNet 或 COCO。

### 4. 相关受益领域与应用
*   **具身智能（Embodied AI）**：对于需要理解三维环境并进行语义理解（如“打开抽屉”、“拿取特定物体”）的机器人至关重要。
*   **工业自动化与仓储机器人**：能够在未经训练的新环境中直接部署，减少生产线配置时间。
*   **家庭服务机器人**：处理非结构化环境（如家庭杂乱桌面）的操作能力将得到质的提升。
*   **边缘计算**：MolmoBot-SPOC 的设计使其能够部署在计算资源受限的边缘设备上，这对于独立运行的移动操作机器人具有极高的实用价值。

### 5. 可推断的局限性
*   **“Sim-to-Real”的最终边界**：尽管摘要强调零样本成功率达 79.2%，但仍有约 20% 的失败率。这暗示了在极度复杂、未见的真实世界物理交互（如极端光照、复杂的接触物理、材质纹理的细微差异）中，模拟器可能仍存在难以弥补的“保真度”限制。
*   **硬件依赖性**：虽然架构通用，但控制策略是否能平滑迁移至具有不同动力学特性（如柔性材料、不同关节摩擦力）的异构机器人平台，仍需进一步验证。
*   **计算成本**：虽然推理可能很快，但构建这样规模的合成数据和训练庞大的 VLM 基础模型，其预训练成本极高，属于资源密集型研究，普通研究者可能难以复现完整数据集的训练过程。

### 专家点评
这篇论文的趣味性在于它**“暴力美学”式的实验设计**——通过将视觉语言模型这一计算机视觉的热点技术与大规模仿真结合，成功绕过了机器人学习中最头疼的“数据匮乏”痛点。对于 CV 研究者而言，**如何利用程序化生成技术弥补现实感知偏差**，是该论文提供的最具启发性的范式。

**Key Findings:**

- With sufficiently large-scale and diverse simulated synthetic training data, we show that zero-shot transfer to the real world is not only possible, but effective for both static and mobile manipulation.
- We introduce MolmoBot-Engine, a fully open-source pipeline for procedural data generation across robots, tasks, and diverse simulated environments in MolmoSpaces.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16861v1)
- [arXiv](https://arxiv.org/abs/2603.16861v1)

---

<a id='2603.18004v1'></a>
## [Unified Spatio-Temporal Token Scoring for Efficient Video VLMs](https://arxiv.org/abs/2603.18004v1)

**Authors:** Jianrui Zhang, Yue Yang, Rohun Tripathi, Winson Han, Ranjay Krishna, Christopher Clark, Yong Jae Lee, Sangho Lee

**Published:** 2026-03-18

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Token pruning is essential for enhancing the computational efficiency of vision-language models (VLMs), particularly for video-based tasks where temporal redundancy is prevalent. Prior approaches typically prune tokens either (1) within the vision transformer (ViT) exclusively for unimodal perception tasks such as action recognition and object segmentation, without adapting to downstream vision-language tasks; or (2) only within the LLM while leaving the ViT output intact, often requiring complex text-conditioned token selection mechanisms. In this paper, we introduce Spatio-Temporal Token Scoring (STTS), a simple and lightweight module that prunes vision tokens across both the ViT and the LLM without text conditioning or token merging, and is fully compatible with end-to-end training. By learning how to score temporally via an auxiliary loss and spatially via LLM downstream gradients, aided by our efficient packing algorithm, STTS prunes 50% of vision tokens throughout the entire architecture, resulting in a 62% improvement in efficiency during both training and inference with only a 0.7% drop in average performance across 13 short and long video QA tasks. Efficiency gains increase with more sampled frames per video. Applying test-time scaling for long-video QA further yields performance gains of 0.5-1% compared to the baseline. Overall, STTS represents a novel, simple yet effective technique for unified, architecture-wide vision token pruning.

**Analysis:**

这是一份关于论文《Unified Spatio-Temporal Token Scoring for Efficient Video VLMs》的深度分析报告。

---

### 1. 摘要翻译
为了提升视觉语言模型（VLM）在处理高时空冗余视频任务时的计算效率，本文提出了“时空标记评分”（STTS）模块。不同于现有方法（仅在ViT或LLM内部进行处理，或依赖复杂的文本条件），STTS是一个简单且轻量的插件，能够直接在ViT内部跨时空修剪冗余标记，且无需文本输入即可端到端训练。通过引入时空双轴评分机制和高效的 packing 算法，STTS 可修剪 50% 的视觉标记，在保持高性能的同时显著提升训练与推理效率。

### 2. 方法动机分析
*   **驱动力**：视频理解中，随着帧数增加，标记序列呈现二次方增长，导致ViT和LLM的计算负担过重。
*   **现有痛点**：预ViT/内ViT方法仅关注空间冗余，忽视跨帧 temporal 冗余；后ViT方法虽然减少了进入LLM的标记数，但无法降低庞大ViT编码器的计算负载。
*   **核心直觉**：利用视频中存在大量背景重复的特性，通过轻量级 scorer 预测并剔除这些“非信息冗余”，且该过程应与模型端到端同步优化。

### 3. 方法设计详解
*   **流程总结**：
    1.  **特征池化**：在ViT中间层（默认第3层）后，通过 $w \times w$ 窗口池化聚合空间特征。
    2.  **双轴评分**：利用三层 MLP 预测标记重要性。输入为当前帧与上一帧拼接特征，捕获跨帧相似性（Temporal）和下游任务感知的空间显著性（Spatial）。
    3.  **偏置注入**：将 log 评分作为偏置项（Bias）注入后续 ViT 层的注意力矩阵中，实现软抑制，并引导端到端梯度更新。
    4.  **硬修剪与Packing**：依据评分剔除 bottom-k% 的标记，并通过“First-Fit Descending”算法将剩余标记打包为稠密 Tensor，以适配 PyTorch 的并行计算要求。
*   **关键公式**：$L = L_{\text{task}} + L_{\text{sim}}$。$L_{\text{sim}}$ 通过 MSE loss 最小化评分与 1-余弦相似度之间的差距，强行引导 scorer 学习剔除相似的重复背景块。

### 4. 方法对比分析
*   **本质区别**：STTS 在 ViT 内部直接修剪，实现了架构全局的标记减少，而非仅仅是对中间输出的截断。
*   **创新点**：引入了显式的邻帧余弦相似度辅助损失函数，使得 scorer 不需文本提示也能精准定位视频中的冗余区域。
*   **适用场景**：适用于任何基于 Patch 的 ViT 编码器架构（如 Molmo, LLaVA），特别是在长视频长上下文的推理场景中优势巨大。

### 5. 实验分析
*   **核心结果**：在 13 个视频 QA 任务上，修剪 50% 的标记仅带来 0.7% 的性能下降，而吞吐量提升高达 62%。
*   **核心结论**：在长视频任务中，结合测试时扩展（Test-Time Scaling）技术，STTS 能在相同计算预算下处理更多帧，性能反而优于基线。
*   **局限**： packing 算法在极小 batch 规模下可能无法充分发挥硬件加速性能。

### 6. 实用指南
*   **开源信息**：GitHub 链接为 `https://github.com/allenai/STTS`。
*   **实现细节**：
    *   **修剪位置**：建议在 ViT 的浅层（如第 3 层）执行，过深修剪会损失太多语义信息。
    *   **辅助 loss**：必须加入余弦相似度监督，否则 LLM 难以单凭任务损失学习到精确的修剪决策。
*   **迁移建议**：该模块作为一个可插拔的插件，仅需修改 ViT 的输出结构和注入注意力 bias，对下游 LLM 没有任何侵入性，易于迁移到其他 VLM 系统。

### 7. 总结
*   **核心思想**：通过时空双轴学习机制，在视频 ViT 内部实时动态修剪冗余标记。
*   **速记版 Pipeline**：
    1. 浅层提取特征并池化；
    2. 利用 MLP 预测时空重要性分数；
    3. 将分数作为偏差融入注意力运算；
    4. 剔除低分标记并重新打包成紧凑序列。

**Key Findings:**

- In this paper, we introduce Spatio-Temporal Token Scoring (STTS), a simple and lightweight module that prunes vision tokens across both the ViT and the LLM without text conditioning or token merging, and is fully compatible with end-to-end training.
- Overall, STTS represents a novel, simple yet effective technique for unified, architecture-wide vision token pruning.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.18004v1)
- [arXiv](https://arxiv.org/abs/2603.18004v1)

---

<a id='2603.18002v1'></a>
## [Loc3R-VLM: Language-based Localization and 3D Reasoning with Vision-Language Models](https://arxiv.org/abs/2603.18002v1)

**Authors:** Kevin Qu, Haozhe Qi, Mihai Dusmanu, Mahdi Rad, Rui Wang, Marc Pollefeys

**Published:** 2026-03-18

**Categories:** cs.CV, cs.AI, cs.CL

**Abstract:**

Multimodal Large Language Models (MLLMs) have made impressive progress in connecting vision and language, but they still struggle with spatial understanding and viewpoint-aware reasoning. Recent efforts aim to augment the input representations with geometric cues rather than explicitly teaching models to reason in 3D space. We introduce Loc3R-VLM, a framework that equips 2D Vision-Language Models with advanced 3D understanding capabilities from monocular video input. Inspired by human spatial cognition, Loc3R-VLM relies on two joint objectives: global layout reconstruction to build a holistic representation of the scene structure, and explicit situation modeling to anchor egocentric perspective. These objectives provide direct spatial supervision that grounds both perception and language in a 3D context. To ensure geometric consistency and metric-scale alignment, we leverage lightweight camera pose priors extracted from a pre-trained 3D foundation model. Loc3R-VLM achieves state-of-the-art performance in language-based localization and outperforms existing 2D- and video-based approaches on situated and general 3D question-answering benchmarks, demonstrating that our spatial supervision framework enables strong 3D understanding. Project page: https://kevinqu7.github.io/loc3r-vlm

**Analysis:**

这份分析报告旨在解构 Loc3R-VLM 论文的核心逻辑与实现机制。

### 1. 摘要翻译
多模态大语言模型（MLLMs）在连接视觉与语言方面取得了显著进展，但在空间理解和视点感知推理方面仍存在困难。现有方法多试图通过几何线索增强输入表示，而非显式教会模型在 3D 空间中进行推理。我们提出了 Loc3R-VLM，这是一个能够从单目视频输入中为 2D 视觉-语言模型赋予高级 3D 理解能力的框架。受人类空间认知的启发，Loc3R-VLM 依赖两个联合目标：通过全局布局重建来构建场景结构的整体表征，以及通过显式情境建模来锚定自我中心视角。这些目标提供了直接的 3D 空间监督，使感知和语言在 3D 上下文中落地。为确保几何一致性和度量尺度对齐，我们利用了从预训练 3D 基础模型中提取的轻量级相机姿态先验。Loc3R-VLM 在基于语言的定位任务上达到了最先进的性能，并在多个 3D 问答基准测试中超越了现有的 2D 和视频方法，证明了我们的空间监督框架能够实现强大的 3D 理解。

### 2. 方法动机分析
*   **驱动力**：解决 MLLM 在处理 3D 空间时的“盲区”问题，即如何从非结构化的 2D 视频中建立起持久的、具备 3D 认知能力的全局映射。
*   **痛点**：现有工作要么依赖昂贵的点云数据（扩展性差），要么仅作为输入增强（本质上未能学会 3D 推理），导致模型缺乏对视点和位置的显式建模。
*   **核心直觉**：人类通过构建“认知地图”（全局结构）和“自我定位”（局部情境）来理解空间，Loc3R-VLM 通过引入两个辅助训练目标模仿了这一过程。

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **相机姿态先验注入**：利用 CUT3R 提取每帧的潜在相机编码，通过可学习投影层将其作为“视觉锚点”预置到视觉 Token 序列中。
    2.  **全局布局重建**：将视觉 patch 映射到 Bird’s-Eye-View (BEV) 平面，通过高斯负对数似然（GNLL）损失监督，迫使模型学习场景的几何布局。
    3.  **情境建模**：引入 `<Pos>` 和 `<Ori>` Token，显式预测代理在 BEV 空间中的坐标和旋转角度，利用包装高斯（Wrapped Gaussian）分布进行旋转监督。
    4.  **端到端训练**：将上述空间监督信号与标准的自回归交叉熵损失（LLM 生成）联合优化。
*   **关键公式**：`L_total = L_CE + λ_BEV * L_BEV + λ_sit * L_sit`。通过平衡三者，使得 LLM 在生成答案时，其隐藏状态已被嵌入空间先验。

### 4. 方法对比分析
*   **本质区别**：不依赖 3D 标注进行推理。相比于将点云作为输入的方法，它仅在训练阶段利用 3D 基础模型作为监督信号，推理阶段仅需普通视频。
*   **创新点**：提出了“空间监督”而非“特征增强”的范式，通过可学习的定位 Token 实现了对自我中心视角的显式控制。

### 5. 实验分析
*   **结论**：在 SQA3D 定位任务上，Acc@1.0m 显著领先；在 3D QA 任务中，对相对方向和路径规划类问题的理解能力提升明显（如 Relative Direction +36.1%）。
*   **局限**：当前的 BEV 投影丢弃了垂直方向细节，对于多楼层或高度敏感的场景表现受限；且受限于 32 帧的输入限制，大场景存在“视野盲区”。

### 6. 实用指南
*   **开源与实现**：基于 LLaVA-Video-7B 开发。关键实现细节是：需先预处理数据集计算 BEV 真值，并将 `<Pos>` 和 `<Ori>` 插入 Token 序列。
*   **迁移建议**：该模块化设计（相机先验 + BEV 重建 + 情境查询）可轻松迁移到其他基于视频的 VLM 架构中，只需保证基础架构能够处理多帧输入即可。

### 7. 总结
*   **核心思想**：通过 BEV 布局重建与显式位置建模，为 2D 视频赋予 3D 空间认知能力。
*   **速记版 Pipeline**：
    1. 提取视频帧的 3D 几何特征先验；
    2. 将图像 Patch 对齐到全局 BEV 空间；
    3. 插入特殊 Token 显式学习代理位置与旋转；
    4. 联合语言和空间损失进行端到端优化。

**Key Findings:**

- We introduce Loc3R-VLM, a framework that equips 2D Vision-Language Models with advanced 3D understanding capabilities from monocular video input.
- Loc3R-VLM achieves state-of-the-art performance in language-based localization and outperforms existing 2D- and video-based approaches on situated and general 3D question-answering benchmarks, demonstrating that our spatial supervision framework enables strong 3D understanding.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.18002v1)
- [arXiv](https://arxiv.org/abs/2603.18002v1)

---

<a id='2603.17995v1'></a>
## [LoST: Level of Semantics Tokenization for 3D Shapes](https://arxiv.org/abs/2603.17995v1)

**Authors:** Niladri Shekhar Dutt, Zifan Shi, Paul Guerrero, Chun-Hao Paul Huang, Duygu Ceylan, Niloy J. Mitra, Xuelin Chen

**Published:** 2026-03-18

**Categories:** cs.CV, cs.GR, cs.LG

**Abstract:**

Tokenization is a fundamental technique in the generative modeling of various modalities. In particular, it plays a critical role in autoregressive (AR) models, which have recently emerged as a compelling option for 3D generation. However, optimal tokenization of 3D shapes remains an open question. State-of-the-art (SOTA) methods primarily rely on geometric level-of-detail (LoD) hierarchies, originally designed for rendering and compression. These spatial hierarchies are often token-inefficient and lack semantic coherence for AR modeling. We propose Level-of-Semantics Tokenization (LoST), which orders tokens by semantic salience, such that early prefixes decode into complete, plausible shapes that possess principal semantics, while subsequent tokens refine instance-specific geometric and semantic details. To train LoST, we introduce Relational Inter-Distance Alignment (RIDA), a novel 3D semantic alignment loss that aligns the relational structure of the 3D shape latent space with that of the semantic DINO feature space. Experiments show that LoST achieves SOTA reconstruction, surpassing previous LoD-based 3D shape tokenizers by large margins on both geometric and semantic reconstruction metrics. Moreover, LoST achieves efficient, high-quality AR 3D generation and enables downstream tasks like semantic retrieval, while using only 0.1%-10% of the tokens needed by prior AR models.

**Analysis:**

以下是对 LoST (Level of Semantics Tokenization) 论文的深度分析：

### 1. 摘要翻译
生成模型中的标记化（Tokenization）是基础技术。然而，针对3D形状的各种最优标记化方案仍未解决。目前最先进的方法主要依赖于为渲染和压缩设计的几何细节层次（LoD）层级，这些方法往往在标记使用上效率低下，且缺乏自回归建模所需的语义连贯性。我们提出了“语义细节层次标记化”（LoST），它根据语义显著性对标记进行排序，使得模型前缀能够解码出具有主体语义的完整、合理的形状，后续标记则用于细化特定实例的几何和语义细节。为了训练LoST，我们引入了“关系间距对齐”（RIDA）这一新型3D语义对齐损失，通过将3D形状潜在空间的关系结构与语义DINO特征空间对齐，实现了高效且高质量的3D生成，且仅需现有模型0.1%-10%的标记量。

### 2. 方法动机分析
- **驱动力**：旨在克服现有3D AR生成模型中标记效率低、语义结构缺失的问题。
- **痛点**：传统的几何LoD（如OctGPT、VertexRegen）强制要求从粗到细进行空间细化，导致“标记膨胀”（coarse scale tokens多），且早期阶段生成的形状语义缺失，生成的中间体不可用。
- **核心直觉**：AR生成应该遵循“先理解整体，后雕琢细节”的认知规律，而非简单地进行空间平铺。

### 3. 方法设计详解
- **LoST Encoder (基于ViT)**：
    - 输入：VAE编码后的3D triplane（平面特征）。
    - 创新点：引入“注册标记”（register tokens $T_R$），这些标记不直接对应空间patch，而是通过带掩码的注意力机制汇总空间特征。
    - 训练技巧：采用**嵌套dropout**和**因果掩码**，强制要求模型必须在短前缀内编码主要信息，从而实现语义级别的层次化。
- **LoST Decoder (生成式DiT)**：
    - 将任务从确定性重建转化为生成式扩散任务，允许在有限信息下进行合理推断，随着前缀增加，生成逐渐过渡到精细重建。
- **RIDA (Relational Inter-Distance Alignment)**：
    - **逻辑**：由于3D形状与2D图像模态不一致，直接回归DINO特征效果不佳。RIDA通过对比学习，要求3D潜在空间中样本点之间的**相对距离关系**与DINO空间一致，通过这种拓扑结构对齐来赋予空间语义。

### 4. 方法对比分析
- **本质区别**：从“空间层次化”（几何）转变为“语义层次化”（认知）。
- **创新点**：
    1. **语义优先的令牌化**：重塑了AR模型的Token流，使得短序列即可获得可用结果。
    2. **RIDA损失**：无需昂贵的渲染过程，即可通过 latent-to-latent 的方式实现语义空间的跨模态对齐。
- **适用场景**：适用于资源受限的3D生成场景，或需要即时预览生成进度的AR工作流。

### 5. 实验分析
- **验证方法**：在1k unseen test shapes上评估重构CD指标、FID及DINO语义相似度。
- **关键结果**：在极低Token预算下（1-16个tokens），LoST在语义指标上大幅超越了OctGPT和VertexRegen。
- **主要优势**：不仅大幅降低了AR生成的Token数量（仅128 tokens），还支持从极短序列开始“语义可见”的增量生成。
- **主要局限**：在高频细节重建上仍依赖较长的Token序列，且目前的AR生成目标长度是固定的。

### 6. 实用指南
- **开源/实现**：项目已开源（lost3d.github.io）。复现关键在于训练阶段的**嵌套dropout策略**，这是确保前缀可用性的关键。
- **迁移性**：该框架是表征无关的（见文中的Trellis实验），可以轻松迁移到其他3D表征（如Gaussian Splats），只需调整输入接口。
- **训练建议**：使用RIDA进行预训练时，务必处理好样本间的相对距离矩阵，确保批次大小（batch size）足够大以进行对比挖掘。

### 7. 总结
- **核心思想**：通过语义显著性重排序Token，实现可即时预览的增量式3D生成。
- **速记版pipeline**：
    1. **特征提取**：将3D形状投影为triplane特征。
    2. **语义注入**：使用RIDA将triplane特征对齐到DINO的语义空间。
    3. **层次编码**：通过带掩码的ViT将空间特征压缩为语义优先的Token流。
    4. **增量解码**：利用DiT模型，根据输入前缀即时生成/精细化3D形状。

**Key Findings:**

- State-of-the-art (SOTA) methods primarily rely on geometric level-of-detail (LoD) hierarchies, originally designed for rendering and compression.
- We propose Level-of-Semantics Tokenization (LoST), which orders tokens by semantic salience, such that early prefixes decode into complete, plausible shapes that possess principal semantics, while subsequent tokens refine instance-specific geometric and semantic details.
- To train LoST, we introduce Relational Inter-Distance Alignment (RIDA), a novel 3D semantic alignment loss that aligns the relational structure of the 3D shape latent space with that of the semantic DINO feature space.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.17995v1)
- [arXiv](https://arxiv.org/abs/2603.17995v1)

---

<a id='2603.17993v1'></a>
## [GMT: Goal-Conditioned Multimodal Transformer for 6-DOF Object Trajectory Synthesis in 3D Scenes](https://arxiv.org/abs/2603.17993v1)

**Authors:** Huajian Zeng, Abhishek Saroha, Daniel Cremers, Xi Wang

**Published:** 2026-03-18

**Categories:** cs.CV, cs.RO

**Abstract:**

Synthesizing controllable 6-DOF object manipulation trajectories in 3D environments is essential for enabling robots to interact with complex scenes, yet remains challenging due to the need for accurate spatial reasoning, physical feasibility, and multimodal scene understanding. Existing approaches often rely on 2D or partial 3D representations, limiting their ability to capture full scene geometry and constraining trajectory precision. We present GMT, a multimodal transformer framework that generates realistic and goal-directed object trajectories by jointly leveraging 3D bounding box geometry, point cloud context, semantic object categories, and target end poses. The model represents trajectories as continuous 6-DOF pose sequences and employs a tailored conditioning strategy that fuses geometric, semantic, contextual, and goaloriented information. Extensive experiments on synthetic and real-world benchmarks demonstrate that GMT outperforms state-of-the-art human motion and human-object interaction baselines, such as CHOIS and GIMO, achieving substantial gains in spatial accuracy and orientation control. Our method establishes a new benchmark for learningbased manipulation planning and shows strong generalization to diverse objects and cluttered 3D environments. Project page: https://huajian- zeng.github. io/projects/gmt/.

**Analysis:**

### 1. 摘要翻译
本文提出了GMT（Goal-Conditioned Multimodal Transformer），一个用于3D场景中6-DOF物体轨迹合成的多模态Transformer框架。该框架通过整合3D边界框几何信息、点云上下文、语义物体类别及目标末端姿态，生成既真实又具备目标导向的轨迹。与传统的仅依赖2D或局部3D表示的方法不同，GMT将轨迹表示为连续的6-DOF姿态序列，并采用一种能够融合几何、语义、上下文及目标信息的定制化条件策略。实验表明，GMT在合成和真实世界基准测试中均优于现有的最优人类动作及人机交互基线，显著提升了空间精度和方向控制能力。

### 2. 方法动机分析
*   **驱动力**：解决机器人交互中的轨迹生成问题，摆脱对特定仿真器或人工策略的依赖，实现跨平台的通用化操作。
*   **现有痛点**：当前研究多为“以人为中心”（Human-centric），将物体视为被动实体；且多依赖2D像素空间或局部3D表示，难以处理复杂的空间几何约束和长程依赖，导致轨迹物理不可行（如穿模、漂移）。
*   **核心假设**：以物体为中心（Object-centric）建模轨迹，并将物体视为动态主体，通过目标状态（Goal）和环境语义几何的多模态融合，可以生成既符合物理规律又具备高通用性的操作路径。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **输入编码**：将历史轨迹、场景点云、物体边界框及动作描述映射为统一的特征空间。
    2.  **多模态特征提取**：
        *   **几何特征**：通过PointNet++处理点云，并将局部几何信息通过距离加权传播到物体边界框上，确保空间意识。
        *   **语义与上下文**：利用CLIP嵌入处理fixture标签和动作描述，通过多头注意力机制对物体边界框进行上下文编码。
    3.  **层次化融合与预测**：采用类似Perceiver的架构，利用可学习的latent array作为信息瓶颈，强制模型进行尺度归一化；最后直接预测连续的6-DOF姿态序列，而非采取传统的解码分段策略。
*   **关键公式**：Eq(1)通过反距离加权（Inverse-distance weighting）将空间特征从点云空间显式传播至物体，这是保证轨迹物理可行性的关键。

### 4. 方法对比分析
*   **本质区别**：摒弃了模拟人类动作来驱动物体的间接范式，转向直接对物体动力学进行建模。
*   **创新贡献**：提出了一种将物体作为“第一类动态实体”的训练框架；建立了空间几何约束优先于语义信息的融合等级，有效降低了碰撞率。
*   **适用场景**：适用于 cluttered 3D 场景下的长程操控任务，尤其在需要跨形态机器人执行相同任务时表现出色。

### 5. 实验分析
*   **验证方法**：在ADT和HD-EPIC数据集上对比了GIMO和CHOIS等基线模型。
*   **关键结论**：GMT在ADE/FDE（空间精度）、FD（轨迹相似度）和AC（方向稳定性）指标上均达到SOTA，且碰撞率明显降低。
*   **局限性**：对场景中的静态fixture annotations依赖度高，且对未见过的环境的泛化能力仍需引入在线推理或强化学习补强。

### 6. 实用指南
*   **开源情况**：项目主页已开放：[https://huajian-zeng.github.io/projects/gmt/](https://huajian-zeng.github.io/projects/gmt/)。
*   **实现细节**：
    *   **预处理**：必须对轨迹进行重采样（200帧）和滤波（排除静止片段），这点对模型收敛至关重要。
    *   **超参数**：$\lambda_{trans}, \lambda_{ori}, \lambda_{rec}, \lambda_{dest}$ 初始化均设为1.0。
*   **迁移能力**：由于输出为通用的6-DOF姿态序列，可通过逆运动学（IK）适配任何机械臂，天然具备跨机器人平台的可移植性。

### 7. 总结
*   **核心思想**：以物体为核心，融合多模态上下文直接生成物理可行的高精度轨迹。
*   **速记版pipeline**：1.提取多模态特征（几何、语义、目标）；2.通过可学习潜变量融合信息；3.预测物体空间坐标序列；4.通过IK驱动机械臂。

**Key Findings:**

- We present GMT, a multimodal transformer framework that generates realistic and goal-directed object trajectories by jointly leveraging 3D bounding box geometry, point cloud context, semantic object categories, and target end poses.
- Extensive experiments on synthetic and real-world benchmarks demonstrate that GMT outperforms state-of-the-art human motion and human-object interaction baselines, such as CHOIS and GIMO, achieving substantial gains in spatial accuracy and orientation control.
- Our method establishes a new benchmark for learningbased manipulation planning and shows strong generalization to diverse objects and cluttered 3D environments.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.17993v1)
- [arXiv](https://arxiv.org/abs/2603.17993v1)

---

<a id='2603.17989v1'></a>
## [Versatile Editing of Video Content, Actions, and Dynamics without Training](https://arxiv.org/abs/2603.17989v1)

**Authors:** Vladimir Kulikov, Roni Paiss, Andrey Voynov, Inbar Mosseri, Tali Dekel, Tomer Michaeli

**Published:** 2026-03-18

**Categories:** cs.CV

**Abstract:**

Controlled video generation has seen drastic improvements in recent years. However, editing actions and dynamic events, or inserting contents that should affect the behaviors of other objects in real-world videos, remains a major challenge. Existing trained models struggle with complex edits, likely due to the difficulty of collecting relevant training data. Similarly, existing training-free methods are inherently restricted to structure- and motion-preserving edits and do not support modification of motion or interactions. Here, we introduce DynaEdit, a training-free editing method that unlocks versatile video editing capabilities with pretrained text-to-video flow models. Our method relies on the recently introduced inversion-free approach, which does not intervene in the model internals, and is thus model-agnostic. We show that naively attempting to adapt this approach to general unconstrained editing results in severe low-frequency misalignment and high-frequency jitter. We explain the sources for these phenomena and introduce novel mechanisms for overcoming them. Through extensive experiments, we show that DynaEdit achieves state-of-the-art results on complex text-based video editing tasks, including modifying actions, inserting objects that interact with the scene, and introducing global effects.

**Analysis:**

这是一篇关于视频生成与编辑领域极具前瞻性的论文。作为该领域的专家，我为您提供以下专业分析：

### 1. 论文核心贡献总结
**DynaEdit** 提出了一种无需训练（training-free）且与模型无关（model-agnostic）的通用视频编辑框架，突破了现有方法在修改视频动作、物体交互及动态事件方面的局限性。该方法通过解决“反转自由（inversion-free）”框架下常见的低频错位和高频抖动问题，使得利用预训练文本到视频（T2V）流模型进行复杂的视频内容与动力学修改成为可能。

### 2. 核心创新与方法论
该论文的创新点在于对现有“反转自由”方法的深度改进：
*   **诊断分析：** 研究者深入探究了直接应用现有方法时导致视频质量下降的根本原因——即**低频错位（global misalignment）**和**高频抖动（high-frequency jitter）**。
*   **抑制机制：** 针对上述问题，论文引入了专门的机制来重新校准视频的动力学，从而在保持原有视频结构的同时，允许模型对动作序列和物体间复杂的交互关系进行语义级编辑，而非简单的画面叠加。
*   **模型无关性：** 由于不干扰预训练模型的内部权重，该方法可以轻松适配不同类型的文本到视频扩散模型（Flow-based models），具备极高的通用性。

### 3. 对计算机视觉领域的潜在影响
*   **突破“编辑自由度”瓶颈：** 传统的视频编辑方法多局限于风格迁移或局部纹理修改，而 DynaEdit 赋予了模型修改“因果关系”和“物理运动”的能力，这是向视频生成领域迈向“可控物理模拟”的重要一步。
*   **降低计算门槛：** 无需训练（training-free）的特性极大地降低了资源需求，使得个人开发者或小型实验室能够利用大型预训练模型进行深度定制化编辑，无需昂贵的微调成本。

### 4. 相关领域与潜在应用
*   **影视后期制作：** 能够直接通过文本指令改变角色动作或引入动态光影效果，极大地简化复杂视觉特效的制作流程。
*   **人机交互与机器人模拟：** 能够生成复杂的交互式视频片段，辅助训练数据增强或进行场景预测。
*   **辅助驾驶与监控分析：** 通过修改交通流中的动态行为，评估算法在罕见或特殊交互场景下的鲁棒性。
*   **创意内容创作：** 为短视频创作、社交媒体增强现实（AR）特效提供强大的语义级编辑工具。

### 5. 可推测的局限性
*   **语义保真度（Semantic Fidelity）：** 虽然名为“无需训练”，但该类方法仍严重依赖于底层流模型（Flow Model）本身的文本理解能力。若原模型对某些动作理解偏差，编辑效果可能会出现语义漂移。
*   **长视频处理能力：** 视频编辑涉及时间维度的一致性，虽然引入了防抖动机制，但在处理超长视频序列时，模型是否会出现随时间推移的“累积偏差”仍有待观察。
*   **物理规律的严谨性：** 该方法通过生成式模型进行编辑，而非物理模拟器。这意味着在处理极端物理交互时（如复杂的碰撞、流体动力学），可能会出现视觉上合理但物理上不严谨的“伪影”。

**专家点评：** 
这篇论文的价值在于它意识到，单纯的图像反转（Inversion）并非视频编辑的唯一路径。通过解决扩散模型在动态生成过程中的动力学不稳定性，DynaEdit 成功地在“保持视频一致性”和“自由度编辑”之间找到了新的平衡点。这为未来视频生成的控制性研究提供了极佳的范式参考。

**Key Findings:**

- Here, we introduce DynaEdit, a training-free editing method that unlocks versatile video editing capabilities with pretrained text-to-video flow models.
- Our method relies on the recently introduced inversion-free approach, which does not intervene in the model internals, and is thus model-agnostic.
- We show that naively attempting to adapt this approach to general unconstrained editing results in severe low-frequency misalignment and high-frequency jitter.
- We explain the sources for these phenomena and introduce novel mechanisms for overcoming them.
- Through extensive experiments, we show that DynaEdit achieves state-of-the-art results on complex text-based video editing tasks, including modifying actions, inserting objects that interact with the scene, and introducing global effects.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.17989v1)
- [arXiv](https://arxiv.org/abs/2603.17989v1)

---

<a id='2603.17980v1'></a>
## [Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding](https://arxiv.org/abs/2603.17980v1)

**Authors:** Shuyao Shi, Kang G. Shin

**Published:** 2026-03-18

**Categories:** cs.CV

**Abstract:**

Recent Multimodal Large Language Models (MLLMs) have shown high potential for spatial reasoning within 3D scenes. However, they typically rely on computationally expensive 3D representations like point clouds or reconstructed Bird's-Eye View (BEV) maps, or lack physical grounding to resolve ambiguities in scale and size. This paper significantly enhances MLLMs with egomotion modality data, captured by Inertial Measurement Units (IMUs) concurrently with the video. In particular, we propose a novel framework, called Motion-MLLM, introducing two key components: (1) a cascaded motion-visual keyframe filtering module that leverages both IMU data and visual features to efficiently select a sparse yet representative set of keyframes, and (2) an asymmetric cross-modal fusion module where motion tokens serve as intermediaries that channel egomotion cues and cross-frame visual context into the visual representation. By grounding visual content in physical egomotion trajectories, Motion-MLLM can reason about absolute scale and spatial relationships across the scene. Our extensive evaluation shows that Motion-MLLM makes significant improvements in various tasks related to 3D scene understanding and spatial reasoning. Compared to state-of-the-art (SOTA) methods based on video frames and explicit 3D data, Motion-MLLM exhibits similar or even higher accuracy with significantly less overhead (i.e., 1.40$\times$ and 1.63$\times$ higher cost-effectiveness, respectively).

**Analysis:**

以下是对论文《Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding》的深度技术分析：

### 1. 摘要翻译
现有的多模态大语言模型（MLLMs）在3D场景空间推理中表现出潜力，但通常依赖昂贵的3D数据（如点云、BEV图）或因缺乏物理基础而无法准确解析绝对尺寸。本文提出了一种名为 **Motion-MLLM** 的新型框架，通过引入与视频同步的惯性测量单元（IMU）采集的自我运动（Egomotion）数据来增强MLLMs。框架包含两个核心模块：（1）级联运动-视觉关键帧过滤模块，利用IMU和视觉特征高效选取代表性关键帧；（2）非对称跨模态融合模块，通过运动Token将自我运动线索和跨帧视觉上下文整合进视觉表示中。评估结果表明，该方法在保持相似或更高精度的同时，相比SOTA方法具有显著更高的成本效益（提升达1.40倍-1.63倍）。

### 2. 方法动机分析
*   **驱动力**：人类理解空间不仅靠视觉，还靠本体感知。作者希望在MLLM中引入类似“本体感知”的运动先验，从而解决仅靠单目视觉难以确立绝对尺度和空间关系的难题。
*   **现有方法痛点**：
    *   显式3D方法（点云/深度图）计算开销极大。
    *   纯2D/视觉方法缺乏物理锚点，无法准确推断物体的真实比例和场景的绝对距离。
*   **研究假设**：通过引入低成本的IMU egomotion数据作为“物理锚点”，可以以轻量级方式建立视觉内容的度量空间感知。

### 3. 方法设计详解
*   **流程Pipeline**：
    1.  **级联过滤（Stage 1-3）**：输入原始视频，Stage 1使用IMU数据进行运动检测，过滤静态/缓慢运动帧；Stage 2进行几何视差检查；Stage 3进行视觉特征相似性分析，仅保留高信息量关键帧。
    2.  **运动编码**：使用GRU（门控循环单元）处理相邻关键帧间的IMU序列，输出代表该时间段运动特性的“运动Token”。
    3.  **非对称融合**：
        *   第一层：双向跨注意力（Bi-Attn），运动Token和视觉Token相互查询，进行特征互补。
        *   第二层：单向跨注意力（Uni-Attn），视觉Token查询运动Token，使运动Token桥接多帧视觉信息，输出“运动增强型视觉Token”。
*   **算法核心**：利用IMU推导出的平移和旋转变化作为帧挑选阈值，结合视觉Token分析，成功将计算开销压缩至原始帧的3%以内。

### 4. 方法对比分析
*   **本质区别**：不依赖昂贵的3D重建或辅助几何编码，而是将IMU数据作为一种物理输入流，通过跨模态注意力机制赋予MLLM绝对尺度感知。
*   **创新贡献**：提出了一种极其轻量级的“运动-视觉”协同过滤策略，不仅降低了处理冗余数据的压力，还通过物理运动路径显式增强了空间关系理解。
*   **适用场景**：所有自带IMU的移动平台（机器人、无人车、智能手机）。

### 5. 实验分析
*   **关键结果**：在VSI-Bench上取得平均分60.3，较 prior SOTA 有+9.6的显著提升。在ScanQA和SQA3D等指标上，仅用4B参数达到甚至超过了更大规模模型的表现。
*   **主要优势**：极高的成本效益，能在较低的算力成本下实现对空间距离和尺寸的准确推理。
*   **主要局限**：对IMU数据质量有一定依赖，且合成IMU数据的过程需要预先进行SfM重建。

### 6. 实用指南
*   **关键实现**：文中使用了Qwen2.5-VL-3B模型作为骨干。训练分为两阶段：先固定视觉骨干训练运动融合模块，再联合微调。
*   **迁移建议**：该方法非常适合迁移到无人机避障、SLAM任务的辅助、以及室内机器人导航中，可作为即插即用的模块增强现有视觉模型的空间感知能力。

### 7. 总结
*   **核心思想**：利用惯性传感器数据作为物理锚点，高效赋予MLLM度量级的3D空间推理能力。
*   **速记版pipeline**：
    1.  利用IMU检测并过滤视频中的冗余低运动帧。
    2.  用GRU提取IMU序列的运动特征。
    3.  通过两层跨注意力机制将运动信息融入视觉Token。
    4.  输入MLLM输出空间认知增强后的答案。

**Key Findings:**

- In particular, we propose a novel framework, called Motion-MLLM, introducing two key components: (1) a cascaded motion-visual keyframe filtering module that leverages both IMU data and visual features to efficiently select a sparse yet representative set of keyframes, and (2) an asymmetric cross-modal fusion module where motion tokens serve as intermediaries that channel egomotion cues and cross-frame visual context into the visual representation.
- Compared to state-of-the-art (SOTA) methods based on video frames and explicit 3D data, Motion-MLLM exhibits similar or even higher accuracy with significantly less overhead (i.e., 1.40$\times$ and 1.63$\times$ higher cost-effectiveness, respectively).

**Links:**

- [PDF](https://arxiv.org/pdf/2603.17980v1)
- [arXiv](https://arxiv.org/abs/2603.17980v1)

---

<a id='2603.17979v1'></a>
## [AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception](https://arxiv.org/abs/2603.17979v1)

**Authors:** Jinho Park, Se Young Chun, Mingoo Seok

**Published:** 2026-03-18

**Categories:** cs.CV

**Abstract:**

Radar is a critical perception modality in autonomous driving systems due to its all-weather characteristics and ability to measure range and Doppler velocity. However, the sheer volume of high-dimensional raw radar data saturates the communication link to the computing engine (e.g., an NPU), which is often a low-bandwidth interface with data rate provisioned only for a few low-resolution range-Doppler frames. A generalized codec for utilizing high-dimensional radar data is notably absent, while existing image-domain approaches are unsuitable, as they typically operate at fixed compression ratios and fail to adapt to varying or adversarial conditions. In light of this, we propose radar data compression with adaptive feedback. It dynamically adjusts the compression ratio by performing gradient descent from the proxy gradient of detection confidence with respect to the compression rate. We employ a zeroth-order gradient approximation as it enables gradient computation even with non-differentiable core operations--pruning and quantization. This also avoids transmitting the gradient tensors over the band-limited link, which, if estimated, would be as large as the original radar data. In addition, we have found that radar feature maps are heavily concentrated on a few frequency components. Thus, we apply the discrete cosine transform to the radar data cubes and selectively prune out the coefficients effectively. We preserve the dynamic range of each radar patch through scaled quantization. Combining those techniques, our proposed online adaptive compression scheme achieves over 100x feature size reduction at minimal performance drop (~1%p). We validate our results on the RADIal, CARRADA, and Radatron datasets.

**Analysis:**

以下是对论文《AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception》的深度技术分析：

### 1. 摘要翻译
雷达在自动驾驶系统中因其全天候特性成为关键感知模态。然而，海量的高维原始雷达数据往往使受限于低带宽（如CAN总线）的通信链路饱和。本文提出了AdaRadar，一种具有自适应反馈机制的雷达数据压缩框架。它通过性能导向的代理梯度下降，动态调整压缩比，并利用零阶梯度近似处理不可导的剪枝与量化操作，避免了在低带宽链路上传输梯度张量。此外，利用雷达特征在频域的稀疏性，通过离散余弦变换（DCT）和缩放量化，实现了超过100倍的特征尺寸压缩，且感知性能损失极小（约1%）。该方法在RADIal、CARRADA和Radatron等数据集上表现出强劲的泛化能力。

### 2. 方法动机分析
*   **驱动力**：高维MIMO雷达产生的原始数据量巨大（单帧可达100MB），严重超出了车辆现有计算架构（如CAN总线）的带宽负载。
*   **现有痛点**：现有的固定压缩比方案（如基于CFAR的点云提取）会损失大量有效信息导致性能退化；而通用的图像压缩方法（如JPEG）未考虑雷达张量的统计特性。
*   **研究假设**：雷达特征图在频域具有强稀疏性，且压缩率应随感知任务的实时需求（而非固定值）动态波动。

### 3. 方法设计详解
*   **流程总结**：
    1.  **分块与频域变换**：将原始雷达张量划分为 $M \times M$ 的块，对每个块进行DCT变换。
    2.  **自适应谱剪枝**：基于目标函数（检测置信度）计算代理梯度，动态决定保留的最高能频域系数个数 $k$。
    3.  **缩放量化**：计算块内峰值 $Q_{c,b}$，以此归一化系数，并进行量化以进一步降维。
    4.  **闭环反馈控制**：在计算端根据检测输出结果，通过零阶优化算法（利用摄动 $\epsilon$）计算代理梯度 $\nabla_{\hat{r}} J$，实时更新剪枝比 $r$，无需后端反向传播。
*   **模型结构**：分为传感器侧（Coder）和计算侧（Decoder）。Coder仅包含轻量级运算（DCT、排序、量化），适合DSP部署；Decoder侧利用感知模型输出作为反馈信号。
*   **算法解释**：关键在于**代理梯度估计**。由于剪枝属于非连续操作，无法直接求导，作者利用 $p - p^-$（扰动前后的置信度差）除以扰动值 $\epsilon$ 来模拟梯度，这种零阶近似规避了反向传播，实现轻量级闭环。

### 4. 方法对比分析
*   **本质区别**：从传统的“静态预设压缩”转变为“任务驱动的动态压缩”。
*   **创新贡献**：提出了一种基于零阶梯度的自适应速率控制机制，不仅实现了高压缩比，还保证了感知任务的鲁棒性。
*   **适用场景**：所有基于FMCW雷达的感知系统，特别是在带宽受限且对精度要求高的自动驾驶场景。

### 5. 实验分析
*   **验证方法**：在RADIal、CARRADA、Radatron三个主流数据集上进行检测和分割任务测试。
*   **关键结果**：在保证下游感知任务精度下降约1%的前提下，实现了超过100倍的数据压缩。
*   **主要优势**：极低的计算复杂度（适合嵌入式），感知精度优于传统点云压缩方案。
*   **主要局限**：若时间维度的关联性较差，且系统在帧间极度剧烈波动，压缩效率可能会受到短暂影响。

### 6. 实用指南
*   **开源情况**：代码及详细配置已在作者主页 `jp4327.github.io/adaradar/` 开源。
*   **实现细节**：建议使用 $M=64$ 作为分块大小以平衡缩放系数开销；初始修剪率 $r_1$ 与学习率 $\eta$ 可根据硬件约束进行微调。
*   **迁移可能**：该机制不仅限于雷达，理论上可迁移到任何需要通过高带宽限制链路传输特征张量的感知任务（如多模态融合边缘计算）。

### 7. 总结
*   **核心思想**：利用感知任务的反馈闭环，实现对雷达频域特征的动态自适应压缩。
*   **速记版Pipeline**：
    1.  对雷达张量进行DCT分块变换。
    2.  基于实时检测精度动态调整剪枝率。
    3.  利用零阶梯度估计进行在线更新。
    4.  对剩余系数进行高效量化并传输。

**Key Findings:**

- In light of this, we propose radar data compression with adaptive feedback.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.17979v1)
- [arXiv](https://arxiv.org/abs/2603.17979v1)

---

<a id='2603.17975v1'></a>
## [AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors](https://arxiv.org/abs/2603.17975v1)

**Authors:** Aymen Mir, Riza Alp Guler, Xiangjun Tang, Peter Wonka, Gerard Pons-Moll

**Published:** 2026-03-18

**Categories:** cs.CV

**Abstract:**

We present AHOY, a method for reconstructing complete, animatable 3D Gaussian avatars from in-the-wild monocular video despite heavy occlusion. Existing methods assume unoccluded input-a fully visible subject, often in a canonical pose-excluding the vast majority of real-world footage where people are routinely occluded by furniture, objects, or other people. Reconstructing from such footage poses fundamental challenges: large body regions may never be observed, and multi-view supervision per pose is unavailable. We address these challenges with four contributions: (i) a hallucination-as-supervision pipeline that uses identity-finetuned diffusion models to generate dense supervision for previously unobserved body regions; (ii) a two-stage canonical-to-pose-dependent architecture that bootstraps from sparse observations to full pose-dependent Gaussian maps; (iii) a map-pose/LBS-pose decoupling that absorbs multi-view inconsistencies from the generated data; (iv) a head/body split supervision strategy that preserves facial identity. We evaluate on YouTube videos and on multi-view capture data with significant occlusion and demonstrate state-of-the-art reconstruction quality. We also demonstrate that the resulting avatars are robust enough to be animated with novel poses and composited into 3DGS scenes captured using cell-phone video. Our project page is available at https://miraymen.github.io/ahoy/

**Analysis:**

以下是对论文《AHOY! Animatable Humans under Occlusion from YouTube Videos》的技术分析：

### 1. 摘要翻译
我们提出AHOY，一种能够从野外单目视频中重建完整、可动画化3D高斯化身的方法，尽管存在严重遮挡。现有方法通常假设输入无遮挡且处于标准姿态，这排除了大量真实世界中存在家具、物体或他人遮挡的视频数据。针对这些挑战，我们的贡献在于：(i) 一种“以幻觉作为监督”的流程，利用身份微调后的扩散模型为未观测的身体区域生成密集监督；(ii) 一种两阶段的“标准到姿态相关”架构，从稀疏观测引导至全姿态相关的高斯图；(iii) 一种地图姿态/LBS姿态解耦策略，用以吸收生成数据中的多视图不一致性；(iv) 一种头/身拆分监督策略以保持面部身份一致性。

### 2. 方法动机分析
*   **核心驱动力**：利用YouTube上数以亿计包含遮挡的视频作为高质量3D人类资产的来源。
*   **痛点**：现有技术（如NeRF/3DGS）极其依赖“完整观测”，导致在处理现实场景中常见的遮挡（如桌子遮挡下半身）时出现严重的空洞、失真或无法建模。
*   **研究假设**：通过视频扩散模型（Video Diffusion）的先验知识可以“填充”被遮挡的身体部位，通过精心设计的“结构化运动”序列可以构建多视角一致性监督。

### 3. 方法设计详解（Pipeline）
1.  **粗略重建 (Coarse Avatar)**：利用DensePose将原始视频的观测像素映射到标准姿态纹理图。对遮挡区域使用FLUX模型进行图像修复（Inpainting），并利用多视图扩散模型生成4个正交视角下的标准姿态图像，训练一个基础的3DGS。
2.  **幻觉监督生成 (Hallucination)**：利用Identity-LoRA对视频扩散模型（Wan 2.2）进行特定人物的身份微调。输入粗略重建的Avatar进行“结构化运动渲染”（如原地转圈），将渲染结果送入RF-Inversion流程，利用模型先验填充缺失纹理，生成高质量的密集监督视频。
3.  **精细化训练 (Full Avatar)**：
    *   **姿态相关高斯图**：使用StyleUNet预测每帧的高斯偏移，从而实现随体态变化的形变（如衣服褶皱）。
    *   **解耦策略**：将输入StyleUNet的pose（Map pose）固定以保证预测的一致性，而将作用于LBS（线性蒙皮）的pose（LBS pose）按帧优化，从而“吸收”扩散模型生成的几何抖动。
    *   **头/身分离**：利用SMPL-to-FLAME映射，将头部监督独立出来，使用专用的多视图面部扩散模型，解决扩散模型生成的人脸身份易变问题。

### 4. 方法对比分析
*   **本质区别**：AHOY并非仅仅依赖输入本身，而是将“视频扩散模型”视为一个生成式先验，通过两阶段优化，将无监督的“补全数据”转化为具有几何一致性的“监督信号”。
*   **创新点**：Map-pose/LBS-pose解耦设计，有效地隔离了扩散模型生成时的几何噪声，这是解决生成式监督导致Avatar“漂移/模糊”的关键。

### 5. 实验分析
*   **验证方法**：在BEHAVE数据集（有真实多视点对照）和YouTube视频上进行验证。
*   **结果**：在定量指标（PSNR, SSIM, LPIPS）上均大幅超越LHM、IDOL等现有单图/视频基线。
*   **局限性**：高度依赖视频扩散模型的质量；推理速度慢，因为包含多次优化迭代。

### 6. 实用指南
*   **开源**：项目官网：[https://miraymen.github.io/ahoy/](https://miraymen.github.io/ahoy/)。
*   **关键步骤**：
    1.  身份微调（LoRA）必须准确，否则后续 hallucination 会导致人物特征改变。
    2.  Masking（掩码）是关键，头部监督与身体监督必须彻底解耦，否则人脸将变成“恐怖谷”效果。
*   **迁移建议**：该“先粗糙重建后通过diffusion refine”的思路可以迁移到任何动态场景建模（如宠物、动态物件）中。

### 7. 总结
*   **核心思想**：通过扩散模型补全遮挡，利用解耦优化保证几何一致性。
*   **速记pipeline**：
    1. 粗建：纹理修复与基础建模。
    2. 幻觉：用微调扩散模型生成补全视频。
    3. 优化：解耦姿态预测与蒙皮形变。
    4. 对齐：头身分离监督确保面部不变。

**Key Findings:**

- We present AHOY, a method for reconstructing complete, animatable 3D Gaussian avatars from in-the-wild monocular video despite heavy occlusion.
- We address these challenges with four contributions: (i) a hallucination-as-supervision pipeline that uses identity-finetuned diffusion models to generate dense supervision for previously unobserved body regions; (ii) a two-stage canonical-to-pose-dependent architecture that bootstraps from sparse observations to full pose-dependent Gaussian maps; (iii) a map-pose/LBS-pose decoupling that absorbs multi-view inconsistencies from the generated data; (iv) a head/body split supervision strategy that preserves facial identity.
- We evaluate on YouTube videos and on multi-view capture data with significant occlusion and demonstrate state-of-the-art reconstruction quality.
- We also demonstrate that the resulting avatars are robust enough to be animated with novel poses and composited into 3DGS scenes captured using cell-phone video.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.17975v1)
- [arXiv](https://arxiv.org/abs/2603.17975v1)

---

<a id='2603.17948v1'></a>
## [VideoAtlas: Navigating Long-Form Video in Logarithmic Compute](https://arxiv.org/abs/2603.17948v1)

**Authors:** Mohamed Eltahir, Ali Habibullah, Yazan Alshoibi, Lama Ayash, Tanveer Hussain, Naeemullah Khan

**Published:** 2026-03-18

**Categories:** cs.CV, cs.AI

**Abstract:**

Extending language models to video introduces two challenges: representation, where existing methods rely on lossy approximations, and long-context, where caption- or agent-based pipelines collapse video into text and lose visual fidelity. To overcome this, we introduce \textbf{VideoAtlas}, a task-agnostic environment to represent video as a hierarchical grid that is simultaneously lossless, navigable, scalable, caption- and preprocessing-free. An overview of the video is available at a glance, and any region can be recursively zoomed into, with the same visual representation used uniformly for the video, intermediate investigations, and the agent's memory, eliminating lossy text conversion end-to-end. This hierarchical structure ensures access depth grows only logarithmically with video length. For long-context, Recursive Language Models (RLMs) recently offered a powerful solution for long text, but extending them to visual domain requires a structured environment to recurse into, which \textbf{VideoAtlas} provides. \textbf{VideoAtlas} as a Markov Decision Process unlocks Video-RLM: a parallel Master-Worker architecture where a Master coordinates global exploration while Workers concurrently drill into assigned regions to accumulate lossless visual evidence. We demonstrate three key findings: (1)~logarithmic compute growth with video duration, further amplified by a 30-60\% multimodal cache hit rate arising from the grid's structural reuse. (2)~environment budgeting, where bounding the maximum exploration depth provides a principled compute-accuracy hyperparameter. (3)~emergent adaptive compute allocation that scales with question granularity. When scaling from 1-hour to 10-hour benchmarks, Video-RLM remains the most duration-robust method with minimal accuracy degradation, demonstrating that structured environment navigation is a viable and scalable paradigm for video understanding.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **VideoAtlas** 的论文分析如下：

### 1. 核心贡献总结
VideoAtlas 提出了一种无需预处理、无损且可扩展的视频分层网格表示方法，旨在解决长视频处理中因过度依赖文本描述（Captioning）而导致的视觉信息丢失问题。通过构建一个递归的层级导航环境，该研究实现了视频理解计算复杂度的“对数级”增长，使得处理 10 小时量级的超长视频成为可能。

### 2. 关键创新与方法论
*   **分层网格表示 (Hierarchical Grid Representation)：** 摒弃了传统的“视频转文本”或“全局帧采样”策略，采用一种统一的、递归的视觉空间结构。这种结构使得系统能够像查看地图一样“缩放”至视频的特定区域或时间切片，从而实现真正的“无损”视觉访问。
*   **视频递归语言模型 (Video-RLM) 与 Master-Worker 架构：** 借鉴递归语言模型（RLM）的思想，将视频理解任务转化为一个马尔可夫决策过程（MDP）。Master 负责全局探索与调度，Workers 负责并发挖掘细节，实现了计算任务在空间和时间上的灵活分配。
*   **计算效率与自适应分配：** 引入了“环境预算（Environment Budgeting）”概念，允许开发者通过限制递归深度来平衡计算开销与精度，同时模型展现出了根据问题粒度自适应分配计算资源的能力。

### 3. 对领域的潜在影响
*   **范式转移：** 该研究挑战了当前依赖“大语言模型（LLM）+ 视频描述器”的传统视频理解范式，转向了更具物理可解释性和计算高效性的“结构化环境导航”范式。
*   **打破长度限制：** 解决超长视频（10小时+）处理中的瓶颈，使模型能够高效处理监控录像、完整电影解析或长时程教育视频，而无需面对线性增长的内存/计算压力。
*   **多模态缓存优化：** 其分层结构天然支持多模态缓存（Multimodal Cache），显著提升了处理重复视频内容的效率（30-60%的缓存命中率），这在工业界大规模视频分析中极具价值。

### 4. 相关应用领域
*   **视频监控与安全：** 高效检索长达数天的监控录像，实现对特定行为的精确定位。
*   **影视媒体分析：** 对长篇电影进行全片剧情索引、风格分析及快速剪辑辅助。
*   **机器人感知与具身智能：** 机器人可以通过该层级结构有效消化长期的环境视频流，构建其“视觉记忆”，从而更高效地规划导航与任务。
*   **医疗影像：** 辅助分析长时程的手术录像或诊断影像，定位病灶或关键手术操作。

### 5. 推断出的潜在局限性
*   **网格构建的初始开销：** 虽然推理是“对数级”的，但将视频转化为这种层级网格结构的初始构建过程是否会产生巨大的计算负担，论文未详述。
*   **递归逻辑的稳定性：** 递归调用（Recursion）在处理复杂逻辑依赖时可能存在“路径错误”，即如果 Master 在高层级调度错误，可能会导致后续的细节挖掘无法挽回地错过目标信息。
*   **对静态与动态内容的适配：** 该结构看起来非常适合静态或平滑变化的视频，但对于极其高频且杂乱的动态内容，层级网格的一致性（Consistency）和语义对齐是否会受到挑战，尚待评估。
*   **训练与推理的偏差：** 这种复杂的导航决策过程通常需要强化学习（RL）训练，相比于端到端的监督学习，其训练稳定性与收敛难度可能更高。

**专家点评：**
VideoAtlas 的趣味性在于它将计算机视觉从“单纯的信号处理”提升到了“空间导航问题”。通过引入对数级增长的复杂度，它为处理海量视觉数据提供了一条可行的工程路径，极有可能成为未来长视频理解领域的一个重要基准架构。

**Key Findings:**

- To overcome this, we introduce \textbf{VideoAtlas}, a task-agnostic environment to represent video as a hierarchical grid that is simultaneously lossless, navigable, scalable, caption- and preprocessing-free.
- We demonstrate three key findings: (1)~logarithmic compute growth with video duration, further amplified by a 30-60\% multimodal cache hit rate arising from the grid's structural reuse.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.17948v1)
- [arXiv](https://arxiv.org/abs/2603.17948v1)

---

