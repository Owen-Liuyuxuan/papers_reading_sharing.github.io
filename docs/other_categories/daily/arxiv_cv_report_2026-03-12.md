time: 20260312

# Arxiv Computer Vision Papers - 2026-03-12

## Executive Summary

### **Arxiv 计算机视觉领域论文日报执行摘要 (2026-03-11)**

**1. 核心主题与趋势**

本日论文集清晰地反映了当前计算机视觉研究的三大融合趋势：

*   **具身智能与机器人学的深度融合**：超过半数论文（1, 2, 5, 6, 7）聚焦于如何将视觉感知转化为物理世界的智能行动。研究重点从“看懂世界”转向“在世界中行动与控制”，涵盖了机器人控制、自动驾驶决策、终身模仿学习以及非结构化环境下的移动操作。
*   **多模态感知与表示的持续演进**：论文在视觉与语言（3）、视觉与触觉（7）、视觉与物理属性（9）的融合上提出了更精细的解决方案。核心挑战从简单的特征对齐，转向解决模态间的**幻觉问题**（3）和建立**不变性与一致性表示**（8），以实现更鲁棒的理解。
*   **基础模型架构与训练范式的专业化适配**：针对点云（4）、神经渲染（9）、扩散模型（10）等特定任务和数据形态，研究者正在设计更轻量、更高效或更物理可解释的专用架构与训练策略，以提升性能与实用性。

**2. 重点论文亮点**

*   **最具系统整合性**：**《DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control》** 提出将视频动态与动作指令在统一的扩散Transformer框架下进行联合建模，为可泛化的机器人控制提供了一个极具潜力的端到端范式。
*   **最具实践创新性**：**《GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations》** 直击大视觉-语言模型（VLMs）中棘手的“计数幻觉”问题，通过引入目标检测的显式 grounding 机制，提供了一种简洁有效的解决方案，对提升VLMs的可靠性有重要价值。
*   **最具技术前沿性**：**《PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction》** 将偏振视觉物理模型与前沿的3D高斯泼溅技术结合，高效解决了反光表面重建这一长期难题，代表了神经渲染向物理精确性发展的重要一步。

**3. 新兴研究方向**

*   **跨模态不变性学习**：如论文8所探讨的，寻找并构建对多模态干扰不变的底层表示，可能成为提升模型鲁棒性的新理论方向。
*   **“终身”与增量学习机制**：论文5提出的终身模仿学习框架，是应对开放世界、非稳态环境的关键，如何高效回放与调整多模态经验是核心挑战。
*   **物理引导的视觉生成与重建**：论文9和10（以语义退化条件引导）均体现了将物理约束或先验（如偏振物理、语义结构）注入数据驱动模型（神经渲染、扩散模型）的强烈趋势，以实现更可控、更符合现实的结果。
*   **轻量化基础模型**：论文4“Pointy”反映了在特定领域（如点云）构建专用、高效基础模型的需求，以平衡性能与计算成本。

**4. 精读建议**

根据研究方向，建议优先阅读：

*   **机器人/自动驾驶研究者**：必读 **Paper 1 (DiT4DiT)** 和 **Paper 2 (DynVLA)**。前者是控制范式的创新，后者专注于复杂场景下的动态推理。
*   **多模态与VLMs研究者**：必读 **Paper 3 (GroundCount)**，它是解决VLM实际缺陷的优秀案例。**Paper 7 (FG-CLTP)** 对具身操作中的细粒度触觉-语言对齐也很有启发性。
*   **3D视觉与神经渲染研究者**：必读 **Paper 9 (PolGS++)**，它在技术融合与解决特定难题上表现突出。
*   **生成模型与基础架构研究者**：建议阅读 **Paper 4 (Pointy)** 了解轻量化Transformer设计，以及 **Paper 10** 关注条件扩散模型的新控制方式。

**总结**：本日论文显示，计算机视觉研究的重心正加速向**可行动的、与物理世界交互的智能系统**迁移，同时追求在多模态理解上更**精确、可靠、符合物理规律**。技术发展呈现出 **“专业化”** 与 **“融合化”** 并行的特点。

---

## Table of Contents

1. [DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control](#2603.10448v1)
2. [DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving](#2603.11041v1)
3. [GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations](#2603.10978v1)
4. [Pointy - A Lightweight Transformer for Point Cloud Foundation Models](#2603.10963v1)
5. [Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](#2603.10929v1)
6. [RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion](#2603.10878v1)
7. [FG-CLTP: Fine-Grained Contrastive Language Tactile Pretraining for Robotic Manipulation](#2603.10871v1)
8. [Beyond Sequential Distance: Inter-Modal Distance Invariant Position Encoding](#2603.10863v1)
9. [PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction](#2603.10801v1)
10. [Guiding Diffusion Models with Semantically Degraded Conditions](#2603.10780v1)

---

## Papers

<a id='2603.10448v1'></a>
## [DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control](https://arxiv.org/abs/2603.10448v1)

**Authors:** Teli Ma, Jia Zheng, Zifan Wang, Chuili Jiang, Andy Cui, Junwei Liang, Shuo Yang

**Published:** 2026-03-11

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models have emerged as a promising paradigm for robot learning, but their representations are still largely inherited from static image-text pretraining, leaving physical dynamics to be learned from comparatively limited action data. Generative video models, by contrast, encode rich spatiotemporal structure and implicit physics, making them a compelling foundation for robotic manipulation. But their potentials are not fully explored in the literature. To bridge the gap, we introduce DiT4DiT, an end-to-end Video-Action Model that couples a video Diffusion Transformer with an action Diffusion Transformer in a unified cascaded framework. Instead of relying on reconstructed future frames, DiT4DiT extracts intermediate denoising features from the video generation process and uses them as temporally grounded conditions for action prediction. We further propose a dual flow-matching objective with decoupled timesteps and noise scales for video prediction, hidden-state extraction, and action inference, enabling coherent joint training of both modules. Across simulation and real-world benchmarks, DiT4DiT achieves state-of-the-art results, reaching average success rates of 98.6% on LIBERO and 50.8% on RoboCasa GR1 while using substantially less training data. On the Unitree G1 robot, it also delivers superior real-world performance and strong zero-shot generalization. Importantly, DiT4DiT improves sample efficiency by over 10x and speeds up convergence by up to 7x, demonstrating that video generation can serve as an effective scaling proxy for robot policy learning. We release code and models at https://dit4dit.github.io/.

**Analysis:**

这是一篇关于具身智能（Embodied AI）领域的创新性论文。以下是对 **DiT4DiT** 方法的深度分析。

---

### 1. 摘要翻译
视觉-语言-动作（VLA）模型虽已成为机器人学习的前沿范式，但其表征往往受限于静态图像-文本预训练，导致物理动力学的学习依赖于有限的动作数据。相比之下，生成式视频模型通过编码丰富的时空结构和隐含物理规律，成为机器人操作的有力基础。为弥合这一鸿沟，我们提出了 **DiT4DiT**，这是一个端到端的视频-动作模型，将视频扩散 Transformer（DiT）与动作扩散 Transformer 耦合在一个统一的级联框架中。DiT4DiT 不依赖于重建未来帧，而是提取视频生成过程中的中间去噪特征，将其作为动作预测的时间基准条件。我们进一步提出了一种具有解耦时间步和噪声尺度的双流匹配（dual flow-matching）目标，实现了视频预测、隐状态提取和动作推理的协同训练。在模拟和真实世界基准测试中，DiT4DiT 达到了 SOTA 水平，在 LIBERO 上取得了 98.6% 的平均成功率，在 RoboCasa-GR1 上取得了 50.8% 的成功率，且训练数据需求大幅降低。它将样本效率提升了 10 倍以上，收敛速度提升了 7 倍，证明了视频生成可作为机器人策略学习的有效缩放代理（scaling proxy）。

---

### 2. 方法动机分析
*   **驱动力**：作者认为机器人控制的瓶颈在于“对物理世界的理解”。静态图像预训练无法捕获物理动态，而视频生成模型天然具备物理先验，因此应将其作为策略学习的“预训练信号”。
*   **现有方法痛点**：
    *   VLA 模型将物理动力学学习完全丢给动作微调阶段，导致极其依赖大规模动作数据（样本效率低）。
    *   现有的多阶段架构（如先生成再控制）在特征提取上存在断层，无法有效利用生成过程中的动态信息。
*   **核心假设**：视频生成中的“中间去噪状态”包含丰富的时空信息，通过显式提取这些中间隐变量并将其作为动作条件，能赋予机器人更强的物理直觉。

---

### 3. 方法设计详解
*   **双 DiT 级联架构**：
    *   **Video DiT**：作为主干网络，通过扩散过程预测未来视觉动态。
    *   **Action DiT**：通过跨注意力（Cross-Attention）机制，从 Video DiT 提取视觉特征，执行动作推理。
*   **关键机制（Tri-timestep scheme）**：
    *   **解耦时间步**：Video DiT 使用均匀采样（捕捉完整物理过程）；Action DiT 使用 Beta 采样（聚焦关键控制阶段）；隐状态提取使用固定时间步 $\tau_f$（确保视觉特征的稳定性）。
    *   **Hook 提取机制**：通过 `forward hook` 在视频生成过程中截取中间层的隐藏激活值，这些特征既包含物理先验，又经过了时间对齐。
*   **双流匹配训练**：将视频预测损失与动作预测损失联合优化，强制视觉 backbone 学习到对动作决策真正“有用”的特征表示，而非单纯的像素重建。

---

### 4. 方法对比分析
*   **本质区别**：与现有模型（如 Cosmos Policy 或 mimic-video）不同，DiT4DiT 不仅仅是将视频模型视为特征提取器，而是通过**联合训练**让两个模块在同一时空分布下协同优化，解决了“特征提取不稳定性”问题。
*   **创新贡献**：提出了一种通用的、端到端的“生成即控制”代理框架，利用视频生成的中间过程特征实现了高频、鲁棒的机器人控制。

---

### 5. 实验分析
*   **验证方法**：在 LIBERO（7-DoF）和 RoboCasa-GR1（29-DoF）模拟器以及真实 Unitree G1 机器人上进行部署。
*   **关键结论**：在零样本（zero-shot）泛化实验中，模型面对未见过的物体（如更换杯子类型、增加物体数量）表现出远超基线的鲁棒性，证明了生成的物理先验具备极强的分布迁移能力。
*   **主要优势**：极高的样本效率（10倍提升），且能够处理长视野（long-horizon）任务，在真实世界中对视觉遮挡的鲁棒性极强。

---

### 6. 实用指南
*   **开源情况**：已发布代码与模型，详见 [dit4dit.github.io](https://dit4dit.github.io/)。
*   **实现细节**：
    *   提取层选择至关重要，实验证明中间层（Layer 18）效果最优。
    *   去噪步数：单次前向传递（1 step）效果最好，过度迭代会导致性能下降。
*   **迁移建议**：该架构适合任何具备 Transformer 主干的视频生成模型，通过在该模型中间层添加 Hook 结构即可迁移至其他动作控制任务。

---

### 7. 总结
*   **核心思想**：利用视频扩散生成的中间隐状态，作为强化机器人动作决策的动力学特征。
*   **速记版 Pipeline**：
    1. 输入观测与目标；
    2. 视频模型开启生成，在特定时间步提取中间视觉特征；
    3. 动作模型读取该特征，结合当前机械臂状态输出动作轨迹；
    4. 两个模型联合训练，使视觉特征不断向动作优化方向对齐。

**Key Findings:**

- To bridge the gap, we introduce DiT4DiT, an end-to-end Video-Action Model that couples a video Diffusion Transformer with an action Diffusion Transformer in a unified cascaded framework.
- Across simulation and real-world benchmarks, DiT4DiT achieves state-of-the-art results, reaching average success rates of 98.6% on LIBERO and 50.8% on RoboCasa GR1 while using substantially less training data.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10448v1)
- [arXiv](https://arxiv.org/abs/2603.10448v1)

---

<a id='2603.11041v1'></a>
## [DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving](https://arxiv.org/abs/2603.11041v1)

**Authors:** Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, Zhaoxiang Zhang, Tieniu Tan

**Published:** 2026-03-11

**Categories:** cs.CV, cs.RO

**Abstract:**

We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT.

**Analysis:**

### 1. 摘要翻译
我们提出了DynVLA，这是一种引入了名为“动力学思维链（Dynamics CoT）”新范式的自动驾驶VLA模型。DynVLA在生成动作前先预测紧凑的世界动力学，从而实现更具信息量和物理基础的决策。为了获得紧凑的动力学表示，DynVLA引入了动力学分词器（Dynamics Tokenizer），将未来演变压缩为一小组动力学标记。针对交互密集型驾驶场景中丰富的环境动态，DynVLA解耦了自我中心（ego-centric）和环境中心（environment-centric）的动力学，从而实现了更准确的世界模型建模。随后，我们通过SFT（监督微调）和RFT（强化微调）训练DynVLA在动作前生成动力学标记，在保持高效推理的同时提高了决策质量。与缺乏精细时空理解的文本CoT以及因密集图像预测导致冗余的视觉CoT相比，动力学CoT以紧凑、可解释且高效的形式捕获了世界演变。在NAVSIM、Bench2Drive和大规模内部数据集上的实验表明，DynVLA始终优于现有的CoT方法，验证了动力学CoT的有效性和实用价值。

### 2. 方法动机分析
*   **驱动力**：旨在解决现有自动驾驶VLA模型在“思维链（CoT）”设计上的困境，即如何在保持推理效率的同时，引入具备时空物理规律的深度推理能力。
*   **现有痛点**：
    *   **文本CoT**：仅在文本空间推理，缺乏对物理世界的细粒度时空理解，导致决策逻辑缺失底层物理支持。
    *   **视觉CoT**：虽能捕获时空关系，但因直接预测像素级未来帧，计算冗余度极高且推理延迟显著。
*   **研究假设**：通过显式建模并压缩未来“动力学”（如物体运动趋势、几何演变）而非预测完整图像或语义文本，可以构建更紧凑、更高效且具备物理因果性的中间推理步骤。

### 3. 方法设计详解
*   **流程总结**：
    1.  **动力学分词器（Dynamics Tokenizer）训练**：将相邻帧图像通过Transformer编码器映射，并利用两组可学习查询（Queries）分别提取“自我中心”和“环境中心”动态。通过VQ-VAE将其离散化为紧凑的动力学标记，并辅以动作相关性正则化（预测自我动作）和跨视图一致性正则化（图像/BEV联合监督），实现解耦的物理空间表示。
    2.  **SFT阶段**：构建包含 `[BOD, 动力学标记, EOD, BOA, 动作标记, EOA]` 的结构化序列，通过预测序列实现从“世界动力学演变”到“具体驾驶动作”的逻辑链。
    3.  **RFT阶段**：利用强化学习（GRPO）优化策略，引入轨迹奖励和格式奖励，强制模型输出符合逻辑的CoT模板，提升规划安全性和长时决策质量。
*   **核心逻辑**：通过将“动力学”作为中间变量，模型实际上在执行一个“预演（Simulation）-决策（Action）”的思维过程。

### 4. 方法对比分析
*   **本质区别**：与现有方法不同，DynVLA通过“动力学压缩”实现时空推理，而非简单的文本描述或像素级生成。
*   **创新点**：
    1.  **解耦动力学建模**：显式分离自我与环境动态，解决了物理模糊性（如 ego 运动与物体运动的混淆）。
    2.  **物理正则化约束**：引入自我动作解码损失和跨视图一致性损失，强制动力学标记具备物理意义。
    3.  **极简推理链**：仅用8个动力学标记即可覆盖未来2秒的复杂演变，实现极高的推理效率。

### 5. 实验分析
*   **验证方法**：在NAVSIM、Bench2Drive及大规模自有数据集上进行基准测试。
*   **结论**：在保持极低推理延迟的前提下，PDMS（路径指标）和碰撞率均显著优于现有SOTA模型（如DriveVLA-W0、FSDrive）。
*   **局限**：在极端天气（如大雨导致的视觉缺失）下，动力学预测仍存在模糊性，导致下游决策不安全。

### 6. 实用指南
*   **关键超参数**：推荐设置 $K=2$（2秒预测视野），动力学标记 $N_{ego}=4, N_{env}=4$。
*   **迁移技巧**：该动力学分词器架构可迁移至任何需要长时预测的机器人操作任务中，核心是保证Encoder输入包含不同时间戳的观测，且 Decoder 需要多任务（如重构）辅助监督。

### 7. 总结
*   **核心思想**：将自动驾驶推理抽象为“物理动力学标记”的预测，实现高效的物理感知决策。
*   **速记版pipeline**：
    1.  **动力学解耦**：将视觉输入拆分为自我/环境两类动力学标记。
    2.  **压缩表示**：通过带约束的VQ分词器将动态演变高度压缩。
    3.  **思维链训练**：先生成紧凑动力学序列，再条件化生成驾驶动作。
    4.  **奖励优化**：利用强化学习对规划轨迹进行安全对齐。

**Key Findings:**

- We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT.
- Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.11041v1)
- [arXiv](https://arxiv.org/abs/2603.11041v1)

---

<a id='2603.10978v1'></a>
## [GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations](https://arxiv.org/abs/2603.10978v1)

**Authors:** Boyuan Chen, Minghao Shao, Siddharth Garg, Ramesh Karri, Muhammad Shafique

**Published:** 2026-03-11

**Categories:** cs.CV, cs.AI

**Abstract:**

Vision Language Models (VLMs) exhibit persistent hallucinations in counting tasks, with accuracy substantially lower than other visual reasoning tasks (excluding sentiment). This phenomenon persists even in state-of-the-art reasoning-capable VLMs. Conversely, CNN-based object detection models (ODMs) such as YOLO excel at spatial localization and instance counting with minimal computational overhead. We propose GroundCount, a framework that augments VLMs with explicit spatial grounding from ODMs to mitigate counting hallucinations. In the best case, our prompt-based augmentation strategy achieves 81.3% counting accuracy on the best-performing model (Ovis2.5-2B) - a 6.6pp improvement - while reducing inference time by 22% through elimination of hallucination-driven reasoning loops for stronger models. We conduct comprehensive ablation studies demonstrating that positional encoding is a critical component, being beneficial for stronger models but detrimental for weaker ones. Confidence scores, by contrast, introduce noise for most architectures and their removal improves performance in four of five evaluated models. We further evaluate feature-level fusion architectures, finding that explicit symbolic grounding via structured prompts outperforms implicit feature fusion despite sophisticated cross-attention mechanisms. Our approach yields consistent improvements across four of five evaluated VLM architectures (6.2--7.5pp), with one architecture exhibiting degraded performance due to incompatibility between its iterative reflection mechanisms and structured prompts. These results suggest that counting failures stem from fundamental spatial-semantic integration limitations rather than architecture-specific deficiencies, while highlighting the importance of architectural compatibility in augmentation strategies.

**Analysis:**

这份论文分析报告旨在为您剖析 GroundCount 框架的核心逻辑及其在视觉语言模型（VLM）计数任务中的贡献。

### 1. 摘要翻译
视觉语言模型（VLMs）在计数任务中表现出持续的幻觉问题，其准确率远低于其他视觉推理任务。尽管现有的推理型 VLM 取得了进展，但这一现象依然存在。相比之下，诸如 YOLO 等 CNN 基础的目标检测模型（ODMs）在空间定位和实例计数方面表现优异且计算开销极低。本文提出了 GroundCount，一个通过利用 ODM 显式空间接地信息来减轻 VLM 计数幻觉的框架。实验表明，在最佳配置下，该方法可将 Ovis2.5-2B 模型的计数准确率提升 6.6 个百分点（达到 81.3%），并因消除了幻觉驱动的推理循环而将推理时间缩短了 22%。消融研究指出，位置编码对强模型有利，而置信度评分通常会引入噪声。此外，结构化提示词（Structured Prompts）在显式符号接地方面的表现优于隐式的特征融合。

### 2. 方法动机分析
*   **驱动力**：解决 VLM 核心架构在处理组合任务（尤其是需要精确空间感知的计数）时的天然短板。
*   **痛点**：VLM 倾向于过度关注文本先验，而忽视视觉令牌，且对于迭代推理模型，难以通过传统的层级向量 steering（转向）来纠正多步计数过程中的幻觉。
*   **研究假设**：计数幻觉并非由 VLM 的推理能力不足导致，而是源于“空间-语义整合”的基本局限，可以通过引入外部高效的符号化空间信息来校准。

### 3. 方法设计详解
GroundCount 包含三种策略（A/B/C）：
*   **Plan A (提示词增强)**：将 ODM 识别到的物体类别、中心位置（离散化为 3x3 网格）、索引和置信度转换为结构化自然语言文本，直接拼接在用户提示词后。这是最直接的接地方式。
*   **Plan B (特征融合)**：通过一个轻量级融合网络，将 ODM 的卷积特征映射与 VLM 的 ViT 补丁（Patch）进行双分支融合。
    *   **双分支结构**：分支 A 使用 FiLM 进行线性调制（缩放与平移），分支 B 利用交叉注意力机制（Cross-Attention）从 CNN 特征中查询信息。两者通过可学习门控结合。
*   **Plan C (混合策略)**：结合 A 与 B，同时利用提示词增强的显式语义和特征融合的隐式空间信息。

### 4. 方法对比分析
*   **本质区别**：与试图修补 VLM 注意力机制的底层 decoding 调整不同，GroundCount 是“即插即用”式的外部辅助，利用了 CNN 在物体定位上的确定性优势。
*   **创新点**：引入了基于 3x3 空间网格的离散化编码方式，使得 ODM 的坐标输出能被 VLM 完美“消化”；同时提出了双分支融合架构（FiLM + Cross-Attention）以解决不同模态特征表示的差异。

### 5. 实验分析（精简版）
*   **结论**：Plan A 在准确率提升和推理效率优化上表现最佳。
*   **关键发现**：在强模型中，位置信息至关重要；而在较弱模型中，过于复杂的空间编码反而会造成干扰。
*   **局限**：对 InternVL3.5-1B 等具有高度自反射机制的模型，结构化提示词可能会破坏其原有的内部推理逻辑，导致性能下降。

### 6. 实用指南
*   **实现要点**：使用 YOLOv13x 获取检测结果；对检测对象进行“从左到右、从下到上”的序列排序（这一步对维护空间一致性至关重要）。
*   **迁移建议**：该方法极易迁移到其他需要空间属性的任务中（如视觉问答的坐标确认）。对于不同架构的 VLM，需通过消融实验决定是否保留置信度和位置编码。

### 7. 总结
*   **核心思想**：利用外部确定性检测器输出，以结构化文本引导 VLM 纠正空间认知偏差。
*   **速记版pipeline**：
    1.  运行目标检测模型获得原始框和置信度；
    2.  将坐标映射到 3x3 网格并按空间顺序排列；
    3.  转化为描述文本并拼接至用户指令；
    4.  VLM 依据增强提示完成推理。

**Key Findings:**

- This phenomenon persists even in state-of-the-art reasoning-capable VLMs. Conversely, CNN-based object detection models (ODMs) such as YOLO excel at spatial localization and instance counting with minimal computational overhead.
- We propose GroundCount, a framework that augments VLMs with explicit spatial grounding from ODMs to mitigate counting hallucinations.
- We further evaluate feature-level fusion architectures, finding that explicit symbolic grounding via structured prompts outperforms implicit feature fusion despite sophisticated cross-attention mechanisms.
- Our approach yields consistent improvements across four of five evaluated VLM architectures (6.2--7.5pp), with one architecture exhibiting degraded performance due to incompatibility between its iterative reflection mechanisms and structured prompts.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10978v1)
- [arXiv](https://arxiv.org/abs/2603.10978v1)

---

<a id='2603.10963v1'></a>
## [Pointy - A Lightweight Transformer for Point Cloud Foundation Models](https://arxiv.org/abs/2603.10963v1)

**Authors:** Konrad Szafer, Marek Kraft, Dominik Belter

**Published:** 2026-03-11

**Categories:** cs.CV, cs.LG

**Abstract:**

Foundation models for point cloud data have recently grown in capability, often leveraging extensive representation learning from language or vision. In this work, we take a more controlled approach by introducing a lightweight transformer-based point cloud architecture. In contrast to the heavy reliance on cross-modal supervision, our model is trained only on 39k point clouds - yet it outperforms several larger foundation models trained on over 200k training samples. Interestingly, our method approaches state-of-the-art results from models that have seen over a million point clouds, images, and text samples, demonstrating the value of a carefully curated training setup and architecture. To ensure rigorous evaluation, we conduct a comprehensive replication study that standardizes the training regime and benchmarks across multiple point cloud architectures. This unified experimental framework isolates the impact of architectural choices, allowing for transparent comparisons and highlighting the benefits of our design and other tokenizer-free architectures. Our results show that simple backbones can deliver competitive results to more complex or data-rich strategies. The implementation, including code, pre-trained models, and training protocols, is available at https://github.com/KonradSzafer/Pointy.

**Analysis:**

以下是针对《Pointy – A Lightweight Transformer for Point Cloud Foundation Models》的深度分析：

### 1. 摘要翻译
点云基础模型的能力近年来大幅提升，通常依赖于从语言或视觉模型进行大规模表征学习。本文采取了一种更为可控的方法，提出了一种轻量级的基于Transformer的点云架构。与过度依赖跨模态监督不同，我们的模型仅在39k个点云上进行训练，却超越了许多在超过200k个训练样本上训练的大型基础模型。令人惊讶的是，我们的方法在性能上接近了那些处理超过百万级点云、图像和文本样本的模型，证明了精心设计的训练方案和架构的价值。为确保严谨的评估，我们进行了全面的复现研究，统一了不同点云架构的训练体系和基准测试。这一实验框架有效隔离了架构选择的影响，凸显了我们设计的优势及其他无需分词器（tokenizer-free）架构的性能，证明了简单的骨干网络同样能带来极具竞争力的结果。

### 2. 方法动机分析
- **驱动力**：旨在剥离大规模跨模态预训练对性能提升的“迷雾”，探究在有限数据下，架构与训练策略本身对模型性能的贡献。
- **痛点**：当前点云领域存在实验设置（数据量、超参数、预处理）不统一的现状，导致难以公平比较架构的优劣；同时，高性能模型往往过度依赖“暴力”跨模态监督，导致推理负担沉重。
- **核心假设**：通过简化架构设计、剔除复杂的跨模态对齐、维持统一的可控训练环境，轻量化Transformer足以达到甚至超越复杂模型的效果。

### 3. 方法设计详解
- **架构Pipeline**：
    1. **Patch Partitioning**：利用最远点采样（FPS）提取锚点，结合k-近邻（kNN）形成局部点集邻域。
    2. **Point Embedding**：类似于PointNet，将原始点云坐标投影至高维空间，通过残差连接保留空间几何信息，并注入可学习的位置编码。
    3. **Hierarchical Transformer**：采用六层Transformer块，利用“块内合并”（Patch Merging）策略逐层减少Token数量，实现多尺度特征提取。
    4. **Token Aggregation**：通过简单的加法合并策略替代复杂的线性映射，在保持几何连通性的同时降低参数量。
- **模型结构**：该架构完全去除了专门的分词器，直接对原始点云进行处理，引入了约3:1的嵌入维度与注意力头比率，旨在增强空间相关性。

### 4. 方法对比分析
- **本质区别**：去除了跨模态监督，仅使用分类目标，且摒弃了复杂的Token生成机制，通过纯净的自监督/分类目标进行预训练。
- **创新贡献**：提出了一种Tokenizer-free的轻量级Transformer基准，通过控制变量实验证明了“简单架构+规范训练”比“复杂模型+非统一数据”更具扩展性和鲁棒性。
- **适用场景**：适用于资源受限的边缘计算场景，以及需要快速在小规模数据上获得强表征的任务。

### 5. 实验分析
- **关键结论**：在仅使用39k训练样本的情况下，Pointy在ModelNet40上达到90.6%的准确率，在ScanObjectNN上达到80.0%，在zero-shot场景下，性能优于多数大规模预训练模型。
- **优势**：训练收敛速度快，参数效率极高（3M-20M），对硬件要求低。
- **局限**：由于仅在分类任务上预训练，对于语义/实例分割等高密度预测任务的泛化能力尚需验证。

### 6. 实用指南
- **开源地址**：[https://github.com/KonradSzafer/Pointy](https://github.com/KonradSzafer/Pointy)
- **实现建议**：
    - 数据预处理：保持坐标归一化至[-1, 1]区间，仅使用z轴旋转增强。
    - 训练细节：AdamW优化器，固定学习率 $1 \times 10^{-4}$，不使用调度器。
- **迁移性**：该结构极易适配到其他三维感知的下游任务，只需替换最终分类头为对应的分割头即可。

### 7. 总结
- **核心思想**：通过纯净的轻量化Transformer架构实现高效的三维表征学习。
- **速记版pipeline**：原始点采样 -> 点集局部映射 -> 层级Transformer聚合 -> 空间特征输出。

**Key Findings:**

- In contrast to the heavy reliance on cross-modal supervision, our model is trained only on 39k point clouds - yet it outperforms several larger foundation models trained on over 200k training samples.
- Interestingly, our method approaches state-of-the-art results from models that have seen over a million point clouds, images, and text samples, demonstrating the value of a carefully curated training setup and architecture.
- Our results show that simple backbones can deliver competitive results to more complex or data-rich strategies.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10963v1)
- [arXiv](https://arxiv.org/abs/2603.10963v1)

---

<a id='2603.10929v1'></a>
## [Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](https://arxiv.org/abs/2603.10929v1)

**Authors:** Fanqi Yu, Matteo Tiezzi, Tommaso Apicella, Cigdem Beyan, Vittorio Murino

**Published:** 2026-03-11

**Categories:** cs.CV, cs.RO

**Abstract:**

We introduce a lifelong imitation learning framework that enables continual policy refinement across sequential tasks under realistic memory and data constraints. Our approach departs from conventional experience replay by operating entirely in a multimodal latent space, where compact representations of visual, linguistic, and robot's state information are stored and reused to support future learning. To further stabilize adaptation, we introduce an incremental feature adjustment mechanism that regularizes the evolution of task embeddings through an angular margin constraint, preserving inter-task distinctiveness. Our method establishes a new state of the art in the LIBERO benchmarks, achieving 10-17 point gains in AUC and up to 65% less forgetting compared to previous leading methods. Ablation studies confirm the effectiveness of each component, showing consistent gains over alternative strategies. The code is available at: https://github.com/yfqi/lifelong_mlr_ifa.

**Analysis:**

这是一份关于论文《Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment》的深度分析报告。

---

### 1. 摘要翻译
本文提出了一种在序列化任务中实现持续策略优化的终身模仿学习框架，该框架在受限的内存和数据条件下运作。我们的方法摒弃了传统的经验回放，完全在多模态潜在空间中运行，通过存储和重用视觉、语言及机器人状态的紧凑表示来支持未来的学习。为进一步稳定适配过程，我们引入了一种增量特征调整（IFA）机制，通过角度边距约束对任务嵌入的演变进行正则化，从而保持任务间的独特性。该方法在 LIBERO 基准测试中确立了新的技术水平，与此前的领先方法相比，AUC 提升了 10–17 个点，遗忘率降低了高达 65%。

### 2. 方法动机分析
*   **驱动力**：旨在解决终身模仿学习（LIL）中的“灾难性遗忘”问题，同时避免依赖繁重的原始数据回放或复杂的生成式模型。
*   **现有痛点**：现有方法要么依赖昂贵的原始数据回放（占用内存大），要么依赖特定任务标识（Task-ID）进行适配，或者在处理新旧任务相似度极高时面临严重的特征漂移（Representation Drift）。
*   **核心假设**：在冻结预训练骨干的前提下，通过在多模态潜在空间中进行特征重放，并利用任务间相似度自适应地拉开特征表示，能有效实现任务间的解耦和持续学习。

### 3. 方法设计详解
*   **多模态潜在重放 (MLR)**：模型不存储原始轨迹数据，而是将视觉、语言和状态等编码后的多模态特征（latent features）存入缓冲区。这不仅极大地节省内存，还减少了对高维度敏感数据的存储需求。
*   **增量特征调整 (IFA)**：这是本文的核心创新。
    *   **原理**：针对新旧任务，引入一个基于角度距离的约束损失。
    *   **公式意义**：$L_{IFA}$ 强制要求当前任务的新特征 $g(T_k)$ 与其专属的语言参考特征 $h^{(r)}(T_k)$ 足够近，同时与旧任务的参考特征 $h^{(r)}(T_j)$ 足够远。
    *   **自适应边距 ($\delta$)**：边距并非固定，而是根据两个参考点之间的角度距离动态缩放，这确保了相似任务受到合理的排斥力，避免了强行推开导致的过度分离。

### 4. 方法对比分析
*   **本质区别**：与需要模型扩容（参数量增加）或任务识别（Task-ID）的方法不同，本方法在推理时完全是“任务不可知（task-agnostic）”的，且不涉及复杂的蒸馏机制，更加轻量。
*   **创新贡献**：将特征调整机制从传统的欧式距离转向“角度空间”，并通过自适应边距控制解决了不同任务间相似度不一的难题。

### 5. 实验分析
*   **验证方法**：在 LIBERO-OBJECT, LIBERO-GOAL, LIBERO-50 基准测试上进行了广泛对比。
*   **关键结论**：在 LIBERO-GOAL 上 AUC 提升显著（从 60.5 到 77.2），且 NBT（负向后向迁移）大幅降低，证明了 IFA 在缓解遗忘方面的卓越性。
*   **优势**：在保持较小内存占用前提下，通过冻结骨干网实现了高效的知识迁移与保留。
*   **局限**：对预训练好的多模态编码器的质量有一定依赖。

### 6. 实用指南
*   **开源情况**：已开源，见 `https://github.com/yfqi/lifelong_mlr_ifa`。
*   **实现细节**：
    *   **关键超参**：$\alpha$ 是控制排斥强度的关键，实验建议在 0.1-0.7 之间根据数据集规模进行微调。
    *   **任务选择**：IFA 不会对所有任务对都生效，仅选择在语言和视角上相似度最高的前 50% 任务对进行正则化。
*   **迁移建议**：该框架天然适用于任何具备多模态特征编码的机器人模仿学习任务。如果特征空间中存在任务重叠（即相似任务），直接套用此 IFA 模块即可获得性能提升。

### 7. 总结
*   **核心思想**：利用多模态潜在空间重放与角度约束，动态实现任务间特征解耦。
*   **速记版pipeline**：
    1. 使用预训练编码器提取多模态状态特征。
    2. 将潜在特征存入内存缓冲区（代替原始数据）。
    3. 训练过程中对比新任务与历史任务的参考嵌入。
    4. 施加自适应角度排斥损失（IFA）以维持任务区分度。

**Key Findings:**

- We introduce a lifelong imitation learning framework that enables continual policy refinement across sequential tasks under realistic memory and data constraints.
- Our approach departs from conventional experience replay by operating entirely in a multimodal latent space, where compact representations of visual, linguistic, and robot's state information are stored and reused to support future learning.
- To further stabilize adaptation, we introduce an incremental feature adjustment mechanism that regularizes the evolution of task embeddings through an angular margin constraint, preserving inter-task distinctiveness.
- Our method establishes a new state of the art in the LIBERO benchmarks, achieving 10-17 point gains in AUC and up to 65% less forgetting compared to previous leading methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10929v1)
- [arXiv](https://arxiv.org/abs/2603.10929v1)

---

<a id='2603.10878v1'></a>
## [RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion](https://arxiv.org/abs/2603.10878v1)

**Authors:** Andrea Patrizi, Carlo Rizzardo, Arturo Laurenzi, Francesco Ruscelli, Luca Rossini, Nikos G. Tsagarakis

**Published:** 2026-03-11

**Categories:** cs.RO

**Abstract:**

We propose a contact-explicit hierarchical architecture coupling Reinforcement Learning (RL) and Model Predictive Control (MPC), where a high-level RL agent provides gait and navigation commands to a low-level locomotion MPC. This offloads the combinatorial burden of contact timing from the MPC by learning acyclic gaits through trial and error in simulation. We show that only a minimal set of rewards and limited tuning are required to obtain effective policies. We validate the architecture in simulation across robotic platforms spanning 50 kg to 120 kg and different MPC implementations, observing the emergence of acyclic gaits and timing adaptations in flat-terrain legged and hybrid locomotion, and further demonstrating extensibility to non-flat terrains. Across all platforms, we achieve zero-shot sim-to-sim transfer without domain randomization, and we further demonstrate zero-shot sim-to-real transfer without domain randomization on Centauro, our 120 kg wheeled-legged humanoid robot. We make our software framework and evaluation results publicly available at https://github.com/AndrePatri/AugMPC.

**Analysis:**

### 1. 摘要翻译
我们提出了一种结合强化学习（RL）与模型预测控制（MPC）的接触显式分层架构。在该架构中，高层RL智能体向底层运动MPC提供步态与导航指令，通过在仿真中进行试错学习，将接触时序的组合优化负担从MPC中卸载。我们证明，仅需极少量的奖励设计与有限的参数调整，即可获得有效的控制策略。我们在跨越50kg至120kg的多个机器人平台上验证了该架构，观察到了在平地及非平地环境下的非周期性步态与时序适应性。所有平台均实现了无需域随机化的零样本仿真到仿真（sim-to-sim）迁移，并在我们120kg的轮足式人形机器人Centauro上实现了无需域随机化的零样本仿真到现实（sim-to-real）迁移。

### 2. 方法动机分析
*   **驱动力**：解决 legged robot 在复杂环境下步态生成困难的问题，通过分层解耦，平衡模型预测控制的物理准确性与强化学习的决策灵活性。
*   **痛点**：传统的基于模型的MPC优化接触序列属于混合整数规划，在线计算复杂度极高；纯端到端RL则难以保证物理约束，且过度依赖域随机化，导致训练低效。
*   **研究假设**：通过将“接触时序调度”分配给RL，将“运动执行与物理约束满足”分配给MPC，可以有效降低模型复杂度，并实现更稳健的零样本迁移。

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **RL 策略层**：以机器人状态、任务目标和MPC健康指数为观测，输出导航指令（基座速度向量）和接触注入动作（决定何时开启新的飞行相位）。
    2.  **接触调度**：RL通过输出动作 `χ_MPC` 动态触发飞行相位的插入，从而生成非周期的接触序列。
    3.  **MPC 执行层**：采用基于DDP（差分动态规划）的轨迹优化，接收高层给出的指令，在当前优化视界内执行刚体动力学控制，保证物理可行性。
    4.  **闭环机制**：采用基于线性RAMP的姿态跟踪与周期性轨迹优化，确保MPC在执行过程中能够处理突发的变化。
*   **关键算法**：RL采用Soft Actor-Critic (SAC) 算法，通过奖励机制鼓励节能（CoT优化）与指令跟踪。MPC利用逆动力学（Inverse Dynamics）公式，在每个节点强制满足刚体动力学约束。

### 4. 方法对比分析
*   **本质区别**：与预定义步态（固定接触模式）或盲目端到端学习（无显式动力学约束）不同，该方法通过在MPC框架内实时插入飞行相，实现了“策略决定的 acyclic（非周期）步态”。
*   **创新贡献**：提出了一种低延迟的接触注入机制，使得RL无需学习具体的电机扭矩，只需学习接触时序的“宏观决策”，显著提升了样本效率。

### 5. 实验分析
*   **验证方法**：在多台机器人（50-120kg）上进行sim-to-sim验证，并在Centauro平台上实测。
*   **结论**：实现了极高的零样本迁移能力；相比纯RL，训练收敛更快且更加稳健；在混合轮足模式下，步态自动适应速度变化，轮式与足式切换流畅。

### 6. 实用指南
*   **开源情况**：代码已开源（https://github.com/AndrePatri/AugMPC）。
*   **实现细节**：
    *   **并行计算**：利用自定义的CPU并行库进行MPC计算，配合GPU环境进行RL训练。
    *   **迁移要点**：需注意MPC的执行延迟补偿（通过调整下一节点参考），这是实现实机控制稳定性的核心。
*   **迁移可能**：该架构通用性强，仅需调整不同平台的URDF和MPC的代价函数权重即可迁移。

### 7. 总结
*   **核心思想**：RL负责调度时序，MPC负责物理约束，两者解耦协同生成非周期步态。
*   **速记版 Pipeline**：
    1. RL 观测状态并给出导航目标与步态触发点。
    2. 系统根据触发指令实时注入飞行相位到 MPC 优化器。
    3. MPC 在有限视界内通过动力学优化生成精确运动指令。
    4. 机器人根据指令实时执行动作并循环往复。

**Key Findings:**

- We propose a contact-explicit hierarchical architecture coupling Reinforcement Learning (RL) and Model Predictive Control (MPC), where a high-level RL agent provides gait and navigation commands to a low-level locomotion MPC.
- We show that only a minimal set of rewards and limited tuning are required to obtain effective policies.
- Across all platforms, we achieve zero-shot sim-to-sim transfer without domain randomization, and we further demonstrate zero-shot sim-to-real transfer without domain randomization on Centauro, our 120 kg wheeled-legged humanoid robot.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10878v1)
- [arXiv](https://arxiv.org/abs/2603.10878v1)

---

<a id='2603.10871v1'></a>
## [FG-CLTP: Fine-Grained Contrastive Language Tactile Pretraining for Robotic Manipulation](https://arxiv.org/abs/2603.10871v1)

**Authors:** Wenxuan Ma, Chaofan Zhang, Yinghao Cai, Guocai Yao, Shaowei Cui, Shuo Wang

**Published:** 2026-03-11

**Categories:** cs.RO

**Abstract:**

Recent advancements in integrating tactile sensing into vision-language-action (VLA) models have demonstrated transformative potential for robotic perception. However, existing tactile representations predominantly rely on qualitative descriptors (e.g., texture), neglecting quantitative contact states such as force magnitude, contact geometry, and principal axis orientation, which are indispensable for fine-grained manipulation. To bridge this gap, we propose FG-CLTP, a fine-grained contrastive language tactile pretraining framework. We first introduce a novel dataset comprising over 100k tactile 3D point cloud-language pairs that explicitly capture multidimensional contact states from the sensor's perspective. We then implement a discretized numerical tokenization mechanism to achieve quantitative-semantic alignment, effectively injecting explicit physical metrics into the multimodal feature space. The proposed FG-CLTP model yields a 95.9% classification accuracy and reduces the regression error (MAE) by 52.6% compared to state-of-the-art methods. Furthermore, the integration of 3D point cloud representations establishes a sensor-agnostic foundation with a minimal sim-to-real gap of 3.5%. Building upon this fine-grained representation, we develop a 3D tactile-language-action (3D-TLA) architecture driven by a flow matching policy to enable multimodal reasoning and control. Extensive experiments demonstrate that our framework significantly outperforms strong baselines in contact-rich manipulation tasks, providing a robust and generalizable foundation for tactile-language-action models.

**Analysis:**

这是一份针对论文《FG-CLTP: Fine-Grained Contrastive Language Tactile Pretraining for Robotic Manipulation》的深度技术分析。

### 1. 摘要翻译
本文提出了一种名为FG-CLTP的细粒度对比语言触觉预训练框架，旨在提升机器人操纵中的触觉感知能力。现有触觉模型多依赖定性描述（如纹理），忽略了精细操纵所需的定量接触状态（如力大小、接触几何、主轴方向）。我们引入了一个包含10万对触觉3D点云与语言描述的数据集（Contact3D），并实现了一种离散数值标记化机制，将物理度量指标注入多模态特征空间。实验表明，FG-CLTP在接触状态分类任务上达到95.9%的准确率，并使回归误差（MAE）降低了52.6%。基于此表示，我们开发了3D-TLA架构，通过流匹配策略实现了高性能的触觉语言动作控制。

### 2. 方法动机分析
*   **驱动力**：旨在解决现有触觉感知模型无法实现“精细操纵”的问题，特别是缺乏对物理参数（力、深度、旋转）的精确理解。
*   **现有方法痛点**：存在“词汇瓶颈”，现有模型仅将触觉锚定为粗糙的定性标签，无法在语义理解与低层动作执行之间建立定量的物理联系。
*   **研究假设**：通过将连续的触觉物理参数离散化为专门的数值Token，可以强制模型在特征空间内学习到具备度量意义的表示。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据构建**：利用仿真环境（TacFlex）生成触觉点云，并结合分析计算提取力、深度、主轴等物理指标。
    2.  **数值标记化（Numeric Tokenization）**：将连续物理量（如深度 2.1mm）离散化为词表Token（如 `<depth_2.1>`），将其加入预训练语言模型的词库。
    3.  **对比学习**：利用CLIP框架，将触觉点云特征与包含上述数值Token的文本描述进行对齐（InfoNCE loss）。
    4.  **辅助回归**：引入MLP回归头，强制触觉编码器输出的特征能还原真实的物理参数（MSE loss）。
    5.  **下游策略（3D-TLA）**：基于Gemma-2B骨干网，利用流匹配（Flow Matching）策略从触觉、视觉和指令中生成动作序列。
*   **核心模块**：对比学习用于全局语义对齐，辅助物理回归用于细粒度特征精炼，两者协同保障了特征的“物理敏感性”。

### 4. 方法对比分析
*   **本质区别**：从“定性语义学习”转向“定量数值物理学习”。
*   **创新贡献**：提出数值Token化策略，打破了触觉学习的词汇瓶颈；构建了首个具备明确物理度量标注的触觉3D数据集。
*   **适用场景**：高精度触觉反馈控制任务，如精密装配、手内操纵及复杂环境下的接触保持。

### 5. 实验分析
*   **验证方法**：通过Contact3D数据集上的分类与回归测试，以及在真实Imeta Y1机械臂上的管件插入、擦拭、书写任务。
*   **关键结果**：在接触状态回归任务中，MAE较SOTA（CLTP）降低了52.6%；真实物理环境下的管件插入任务成功率提升至85%。
*   **主要优势**：高精度的物理量感知能力；强大的跨传感器泛化性能（sim2real gap仅3.5%）。
*   **主要局限**：对离散化粒度（Bin size）敏感，过粗则精度下降，过细则增大词表规模。

### 6. 实用指南
*   **开源情况**：数据集Contact3D已开源。
*   **实现细节**：
    *   **预处理**：需统一不同传感器的坐标空间至标准范围。
    *   **训练策略**：分两阶段训练，先冻结主干训练MLP，再使用LoRA进行联合微调以防止遗忘。
*   **迁移可能**：该方法中“数值标记化”思路可直接迁移至任何涉及连续标量（如视觉距离、位姿）的VLA模型任务中。

### 7. 总结
*   **核心思想**：将连续物理量数值化为Token，实现触觉感知的定量语义对齐。
*   **速记版pipeline**：
    1. 计算触觉物理量并将其转化为数值Token。
    2. 将触觉点云与包含数值Token的文本对齐。
    3. 训练回归头强制模型感知精细物理特征。
    4. 接入大模型利用流匹配执行操纵任务。

**Key Findings:**

- To bridge this gap, we propose FG-CLTP, a fine-grained contrastive language tactile pretraining framework.
- We first introduce a novel dataset comprising over 100k tactile 3D point cloud-language pairs that explicitly capture multidimensional contact states from the sensor's perspective.
- The proposed FG-CLTP model yields a 95.9% classification accuracy and reduces the regression error (MAE) by 52.6% compared to state-of-the-art methods.
- Building upon this fine-grained representation, we develop a 3D tactile-language-action (3D-TLA) architecture driven by a flow matching policy to enable multimodal reasoning and control.
- Extensive experiments demonstrate that our framework significantly outperforms strong baselines in contact-rich manipulation tasks, providing a robust and generalizable foundation for tactile-language-action models.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10871v1)
- [arXiv](https://arxiv.org/abs/2603.10871v1)

---

<a id='2603.10863v1'></a>
## [Beyond Sequential Distance: Inter-Modal Distance Invariant Position Encoding](https://arxiv.org/abs/2603.10863v1)

**Authors:** Lin Chen, Bolin Ni, Qi Yang, Zili Wang, Kun Ding, Ying Wang, Houwen Peng, Shiming Xiang

**Published:** 2026-03-11

**Categories:** cs.CV

**Abstract:**

Despite the remarkable capabilities of Multimodal Large Language Models (MLLMs), they still suffer from visual fading in long-context scenarios. Specifically, the attention to visual tokens diminishes as the text sequence lengthens, leading to text generation detached from visual constraints. We attribute this degradation to the inherent inductive bias of Multimodal RoPE, which penalizes inter-modal attention as the distance between visual and text tokens increases. To address this, we propose inter-modal Distance Invariant Position Encoding (DIPE), a simple but effective mechanism that disentangles position encoding based on modality interactions. DIPE retains the natural relative positioning for intra-modal interactions to preserve local structure, while enforcing an anchored perceptual proximity for inter-modal interactions. This strategy effectively mitigates the inter-modal distance-based penalty, ensuring that visual signals remain perceptually consistent regardless of the context length. Experimental results demonstrate that by integrating DIPE with Multimodal RoPE, the model maintains stable visual grounding in long-context scenarios, significantly alleviating visual fading while preserving performance on standard short-context benchmarks. Code is available at https://github.com/lchen1019/DIPE.

**Analysis:**

### 1. 摘要翻译
尽管多模态大语言模型（MLLMs）功能强大，但在长上下文场景中仍会面临“视觉衰减”问题。即随着文本序列变长，模型对视觉标记的关注度降低，导致生成的文本脱离视觉约束。我们认为这种退化归因于多模态 RoPE 的内在归纳偏置，即随着视觉和文本标记间距离的增加，模型会惩罚跨模态注意力。为此，我们提出了跨模态距离不变位置编码（DIPE），这是一种简单有效的机制，根据模态交互解耦位置编码。DIPE 在保留模态内交互的自然相对位置以维持局部结构的同时，强制跨模态交互遵循锚定的感知邻近度。该策略有效地缓解了基于距离的跨模态惩罚，确保无论上下文长度如何，视觉信号都能保持感知的连贯性。实验结果表明，通过将 DIPE 与多模态 RoPE 集成，模型在长上下文场景中保持了稳定的视觉定位，显著缓解了视觉衰减，同时在标准短上下文基准测试中保持了性能。

### 2. 方法动机分析
*   **驱动力**：旨在解决多模态大模型在长文本生成中对初始视觉信息“遗忘”或“模糊”的问题。
*   **现有方法痛点**：当前主流的多模态位置编码（如 MRoPE）将视觉和文本统一在同一序列框架下，并沿用 RoPE 的长程衰减机制。随着生成文本增多，视觉标记与新生成文本的相对距离单调增长，导致注意力衰减，造成视觉约束失效。
*   **研究假设**：视觉信息应具有“感知恒定性”。即无论对话生成多长，视觉对象在人类感知中始终是“当前的”，而非像陈旧文本那样逐渐远离。因此，跨模态注意力不应随序列距离而发生强烈的指数级衰减。

### 3. 方法设计详解
*   **流程总结**：
    1.  **模态交互解耦**：将注意力机制分为模态内（Intra-modal）与跨模态（Inter-modal）两部分。
    2.  **模态内注意力（SPE）**：沿用标准的 MRoPE，保留空间和顺序结构信息，处理文本-文本或视觉-视觉交互。
    3.  **跨模态注意力（APE）**：引入“锚定位置编码”。对于文本侧的查询（Query），强制将其位置索引锚定在所属模态分段（Segment）的起始位置；对于视觉侧的键（Key），保持原始相对位置。
    4.  **动态融合**：通过 LogSumExp 统计量，将两个并行的注意力计算结果进行平滑融合（公式化表示为两个子核的加权和）。
*   **算法核心**：通过 `LogSumExp` 技巧合并输出，利用 `sigmoid` 函数动态调整模态内与跨模态注意力权重，无需引入额外参数，且与 FlashAttention 完全兼容。

### 4. 方法对比分析
*   **本质区别**：传统方法采用单一的、基于绝对索引的相对距离度量；DIPE 采用“双视角”机制，即对于跨模态交互，刻意忽略绝对距离增长，强制拉近视觉与文本的感知距离。
*   **创新贡献**：提出了“锚定位置编码”（APE）概念，巧妙解决了长上下文中跨模态注意力随距离衰减的结构性矛盾。
*   **适用场景**：所有涉及长文本生成、长视频理解以及多图交互的 MLLM 场景。

### 5. 实验分析
*   **验证方法**：通过构建“长上下文 VQA 协议”，人为插入大量文本干扰（1K-32K tokens），模拟视觉 fading 现象进行压力测试。
*   **关键结果**：在长上下文场景中，DIPE 带来平均 4.10% 的准确率提升；在轻量化模型（0.5B）上提升更为显著（8.81%）。
*   **优缺点**：有效解决了视觉 fading，且不损害短上下文性能；主要局限在于需要双核计算（虽然不增加总 FLOPs，但对算子实现有一定要求）。

### 6. 实用指南
*   **实现细节**：需在 `attn_forward` 中维护两套 Query（SPE 和 APE 编码），并使用 LogSumExp 进行融合。
*   **迁移可能**：该方法本质是位置编码的处理策略，可轻松迁移至任何基于 RoPE 或 MRoPE 的多模态架构（如 LLaVA, Qwen-VL 等）。

### 7. 总结
*   **核心思想**：通过模态感知解耦，强制跨模态注意力保持距离不变性。
*   **速记版 pipeline**：
    1. 识别文本与视觉的分段；
    2. 计算基于原始位置的 SPE Query；
    3. 计算基于分段起始点的 APE Query；
    4. 并行执行注意力计算；
    5. 利用 LogSumExp 融合两路注意力输出。

**Key Findings:**

- To address this, we propose inter-modal Distance Invariant Position Encoding (DIPE), a simple but effective mechanism that disentangles position encoding based on modality interactions.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10863v1)
- [arXiv](https://arxiv.org/abs/2603.10863v1)

---

<a id='2603.10801v1'></a>
## [PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction](https://arxiv.org/abs/2603.10801v1)

**Authors:** Yufei Han, Chu Zhou, Youwei Lyu, Qi Chen, Si Li, Boxin Shi, Yunpeng Jia, Heng Guo, Zhanyu Ma

**Published:** 2026-03-11

**Categories:** cs.CV

**Abstract:**

Accurate reconstruction of reflective surfaces remains a fundamental challenge in computer vision, with broad applications in real-time virtual reality and digital content creation. Although 3D Gaussian Splatting (3DGS) enables efficient novel-view rendering with explicit representations, its performance on reflective surfaces still lags behind implicit neural methods, especially in recovering fine geometry and surface normals. To address this gap, we propose PolGS++, a physically-guided polarimetric Gaussian Splatting framework for fast reflective surface reconstruction. Specifically, we integrate a polarized BRDF (pBRDF) model into 3DGS to explicitly decouple diffuse and specular components, providing physically grounded reflectance modeling and stronger geometric cues for reflective surface recovery. Furthermore, we introduce a depth-guided visibility mask acquisition mechanism that enables angle-of-polarization (AoP)-based tangent-space consistency constraints in Gaussian Splatting without costly ray-tracing intersections. This physically guided design improves reconstruction quality and efficiency, requiring only about 10 minutes of training. Extensive experiments on both synthetic and real-world datasets validate the effectiveness of our method.

**Analysis:**

### 1. 摘要翻译
准确重建反射表面是计算机视觉中的一个基本挑战。尽管3D高斯溅射（3DGS）实现了高效的新视角渲染，但由于缺乏对反射表面的结构化约束，其在几何和表面法线恢复方面仍落后于隐式神经表示方法。为此，我们提出了PolGS++，一种物理引导的极化高斯溅射框架，用于快速反射表面重建。具体而言，我们将极化双向反射分布函数（pBRDF）集成到3DGS中，明确解耦漫反射和镜面反射分量，为反射表面恢复提供物理基础建模和几何线索。此外，我们引入了一种深度引导的可见性掩码获取机制，实现了高斯溅射中基于极化角（AoP）的切空间一致性约束，无需昂贵的光线追踪计算。该设计在保持高效的同时，仅需约10分钟的训练即可显著提升重建质量。

### 2. 方法动机分析
*   **驱动力**：旨在解决现有3DGS在处理高反射、少纹理表面时几何恢复不稳定、法线精度低的问题，同时克服隐式SDF方法耗时极长的瓶颈。
*   **现有痛点**：3DGS缺乏对物体表面几何结构的直接约束（法线主要依赖像素级光度一致性），难以应对反射带来的视角依赖性伪影；而基于SDF的极化方法通常涉及复杂的体渲染优化，训练极为缓慢。
*   **核心假设**：利用极化信息（pBRDF）提供的物理约束，结合多视角几何一致性，可以有效弥补显式表示（高斯）在处理反射表面时的几何歧义。

### 3. 方法设计详解
*   **Pipeline总结**：
    1.  **pBRDF解耦**：利用极化相机捕捉的Stokes向量，将物体外观分解为漫反射（由3DGS建模）和镜面反射（由CubeMap编码器建模）。
    2.  **物理渲染**：通过pBRDF模型将上述分量融合，并添加极化损失约束。
    3.  **切空间一致性（TSC）**：利用AoP信息约束法线，解决方位角模糊问题。
    4.  **深度引导掩码**：为了在3DGS中引入TSC而无需实时射线追踪，通过比较渲染深度与投影几何距离，判断某点是否在另一视角下可见，从而动态计算 visibility mask。
*   **关键公式意义**：
    *   式(19)是极化渲染方程，将漫反射与镜面反射通过Mueller矩阵进行加权叠加，实现了对极化信息的物理建模。
    *   式(22)是深度引导的可见性判断，通过 $\tau$ 阈值平滑处理高斯溅射的深度不确定性，实现了对不可见区域的有效剔除。

### 4. 方法对比分析
*   **本质区别**：与传统SDF方法对比，该方法保持了3DGS的快速渲染特性；与普通3DGS对比，该方法通过引入极化物理先验和几何约束，大幅提升了法线重建的鲁棒性。
*   **创新点**：首次将多视角切空间一致性（TSC）损失引入3DGS架构，并通过深度引导的轻量化掩码获取策略解决了可见性计算的效率瓶颈。

### 5. 实验分析
*   **关键结果**：在合成与真实数据集上，PolGS++在保持与SDF方法相当的几何重建质量的同时，训练速度实现了约80倍的加速。
*   **优势**：极化信息能够极好地解决少纹理物体的几何重建；深度引导掩码策略高效且鲁棒。
*   **局限**：对极化测量的精度高度依赖（校准误差敏感）；目前仅适用于电介质材料；无法处理动态光源环境。

### 6. 实用指南
*   **开源**：https://github.com/PRIS-CV/PolGS_plus
*   **注意细节**：训练需“热身”阶段，即前1000次迭代仅优化基础几何，随后加入极化损失和延迟渲染；阈值 $\tau=0.010$ 是平衡效果与稳定性的关键超参数。
*   **迁移建议**：深度引导可见性掩码策略可直接迁移至其他基于多视角的3DGS几何约束任务中，以替代耗时的光线追踪。

### 7. 总结
*   **核心思想**：利用极化物理先验与深度感知掩码提升高斯溅射的反射表面重建质量。
*   **速记版pipeline**：
    1. 极化外观物理拆解。
    2. 渲染深度与距离判别可见性。
    3. 利用极化信息约束法线。
    4. 联合损失优化几何表面。

**Key Findings:**

- Although 3D Gaussian Splatting (3DGS) enables efficient novel-view rendering with explicit representations, its performance on reflective surfaces still lags behind implicit neural methods, especially in recovering fine geometry and surface normals.
- To address this gap, we propose PolGS++, a physically-guided polarimetric Gaussian Splatting framework for fast reflective surface reconstruction.
- Furthermore, we introduce a depth-guided visibility mask acquisition mechanism that enables angle-of-polarization (AoP)-based tangent-space consistency constraints in Gaussian Splatting without costly ray-tracing intersections.
- Extensive experiments on both synthetic and real-world datasets validate the effectiveness of our method.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10801v1)
- [arXiv](https://arxiv.org/abs/2603.10801v1)

---

<a id='2603.10780v1'></a>
## [Guiding Diffusion Models with Semantically Degraded Conditions](https://arxiv.org/abs/2603.10780v1)

**Authors:** Shilong Han, Yuming Zhang, Hongxia Wang

**Published:** 2026-03-11

**Categories:** cs.CV

**Abstract:**

Classifier-Free Guidance (CFG) is a cornerstone of modern text-to-image models, yet its reliance on a semantically vacuous null prompt ($\varnothing$) generates a guidance signal prone to geometric entanglement. This is a key factor limiting its precision, leading to well-documented failures in complex compositional tasks. We propose Condition-Degradation Guidance (CDG), a novel paradigm that replaces the null prompt with a strategically degraded condition, $\boldsymbol{c}_{\text{deg}}$. This reframes guidance from a coarse "good vs. null" contrast to a more refined "good vs. almost good" discrimination, thereby compelling the model to capture fine-grained semantic distinctions. We find that tokens in transformer text encoders split into two functional roles: content tokens encoding object semantics, and context-aggregating tokens capturing global context. By selectively degrading only the former, CDG constructs $\boldsymbol{c}_{\text{deg}}$ without external models or training. Validated across diverse architectures including Stable Diffusion 3, FLUX, and Qwen-Image, CDG markedly improves compositional accuracy and text-image alignment. As a lightweight, plug-and-play module, it achieves this with negligible computational overhead. Our work challenges the reliance on static, information-sparse negative samples and establishes a new principle for diffusion guidance: the construction of adaptive, semantically-aware negative samples is critical to achieving precise semantic control. Code is available at https://github.com/Ming-321/Classifier-Degradation-Guidance.

**Analysis:**

### 1. 摘要翻译
分类器无关引导（CFG）是现代文本生成图像模型的核心，但其对语义空泛的空提示符（∅）的依赖导致引导信号容易产生几何纠缠，限制了模型在复杂组合任务中的精度。为此，我们提出了“条件退化引导”（CDG），这是一种通过用策略性退化的条件（$c_{\text{deg}}$）替代空提示符的新范式。这种方法将引导从粗糙的“好 vs. 空”对比转化为更精细的“好 vs. 近乎好”辨别，从而强迫模型捕获细粒度的语义区别。我们发现Transformer文本编码器中的token分为编码对象语义的“内容token”和捕获全局上下文的“上下文聚合token”。通过仅对前者进行分层退化，CDG无需外部模型或额外训练即可构建$c_{\text{deg}}$。在Stable Diffusion 3、FLUX和Qwen-Image等架构上的验证表明，CDG显著提升了组合准确度和图文对齐度，且仅引入了极小的计算开销。

### 2. 方法动机分析
- **驱动力**：旨在解决CFG中由于空提示符（$\emptyset$）与正向条件（$c$）语义差距过大，导致引导信号发生“几何纠缠”，进而引发生成伪影或组合逻辑失效的问题。
- **痛点**：现有改进方法（如修正已有信号或使用外部模型）要么是“头痛医头”的后处理，要么需要引入外部模型（如VLM或弱模型），增加了复杂度和计算成本。
- **研究假设**：Transformer编码器内部存在天然的结构分层——“内容token”（细节）与“上下文聚合token”（结构/风格），通过利用这种语义结构进行受控退化，可以实现更纯净、更正交的引导信号。

### 3. 方法设计详解
CDG通过三个核心步骤实现：
1. **token重要性分析（WPR）**：利用Weighted PageRank算法在Transformer的自注意力图中构建token关系图，计算每个token的语义重要性。该步骤明确了哪些token携带核心语义（Content），哪些仅提供全局支持（CtxAgg）。
2. **分层退化（Stratified Degradation）**：根据设定的退化比率$R_{\text{deg}}$，将token集划分为“内容集”和“上下文聚合集”。优先退化重要性高的内容token（$R_{\text{deg}} \in [0, 1.0]$），再退化其余部分（$R_{\text{deg}} \in (1.0, 2.0]$）。
3. **掩码插值（Masked Interpolation）**：通过二进制掩码$m$，在原始条件$c$和空提示符$\emptyset$之间进行加权生成退化条件$c_{\text{deg}}$：$c_{\text{deg}} = m \odot c + (1 - m) \odot \emptyset$。

### 4. 方法对比分析
- **本质区别**：不同于传统的“全或无”对比，CDG利用自身token结构的语义冗余，通过在输入空间构造“语义相似的负样本”，实现了更平滑、更精准的负向引导。
- **创新贡献**：提出了一种基于Transformer内部结构的通用退化策略，无需训练、插件化、轻量级且对模型架构不敏感（验证了SD3、SD3.5、FLUX等）。
- **适用场景**：极度依赖组合逻辑的任务（如空间位置绑定、多属性描述、文字渲染）。

### 5. 实验分析
- **验证方法**：在MS-COCO和GenAI-Bench上，对比CFG及多种主流改进基线（如PAG, ICG, CADS）。
- **关键结果**：在组合技能（计数、对比、空间关系）上表现显著优于基线；通过几何指标分析（Decoupling/Interference），证明CDG生成的信号更正交于去噪空间。
- **局限**：在某些非常简单的生成任务上，提升边际效益可能不明显。

### 6. 实用指南
- **开源信息**：项目代码已开源（github.com/Ming-321/Classifier-Degradation-Guidance）。
- **实现细节**：建议使用默认参数$R_{\text{deg}} = 1.0$，此时模型会通过“掩码直接替换”跳过复杂的WPR计算，实现极低开销。
- **迁移性**：由于利用了Transformer通用的Attention机制，该方法极易迁移至任何基于Transformer的文本生成图像模型（如Stable Diffusion, FLUX, 以及各类LLM驱动的图像生成器）。

### 7. 总结
- **核心思想**：利用语义分层，构造接近真实的“近乎好”负样本以实现精准引导。
- **速记版pipeline**：
  1. 计算：提取自注意力图并运行WPR得出token重要性。
  2. 划分：根据重要性排名，将token分为语义实体与全局上下文。
  3. 掩码：依比例对内容token进行语义擦除。
  4. 引导：以退化条件作为CFG负提示符进行扩散去噪。

**Key Findings:**

- We propose Condition-Degradation Guidance (CDG), a novel paradigm that replaces the null prompt with a strategically degraded condition, $\boldsymbol{c}_{\text{deg}}$.
- Our work challenges the reliance on static, information-sparse negative samples and establishes a new principle for diffusion guidance: the construction of adaptive, semantically-aware negative samples is critical to achieving precise semantic control.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.10780v1)
- [arXiv](https://arxiv.org/abs/2603.10780v1)

---

