time: 20260325

# Arxiv Computer Vision Papers - 2026-03-25

## Executive Summary

### **Arxiv 计算机视觉领域论文日报执行摘要 (2026-03-23)**

**1. 核心主题与趋势**

今日论文集清晰地反映了计算机视觉领域的三个主要演进方向：

*   **智能体与具身智能的融合**：多篇论文聚焦于如何让AI系统（智能体）在物理或动态世界中感知、推理和行动。这超越了传统的静态图像理解，迈向与环境和任务交互的“具身智能”（`CaP-X`, `WildWorld`, `AgentRVOS`）。
*   **效率与自适应的生成模型**：生成模型（尤其是扩散模型）的研究重点从追求纯粹的质量转向**计算效率**和**内容自适应**。研究者们致力于在保持高质量输出的同时，大幅降低计算成本（`Foveated Diffusion`, `VISion On Request`），或使模型能根据特定需求（如修复、运动估计）进行自适应生成（`DA-Flow`, `TETO`）。
*   **三维场景理解的泛化与简化**：三维视觉研究继续追求两个目标：一是构建更**通用、无约束**的3D场景表示（`OccAny`），二是探索如何用**更少的数据（如单视图）** 完成复杂的3D任务（如新视图生成 `One View Is Enough!`），旨在降低应用门槛。

**2. 重点与创新性论文**

*   **`CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation`**：**具身智能领域的基石性工作**。它不仅提出了一个用于机器人操作的代码智能体基准，更提供了一个系统性框架来评估和改进这类智能体。这对于标准化和加速机器人学习与规划研究具有重要意义。
*   **`Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation`**：**生成模型效率优化的杰出代表**。借鉴人眼“中央凹视觉”原理，对图像/视频不同区域进行差异化计算，在几乎不损失感知质量的前提下实现显著加速。这是将经典计算机视觉思想与前沿生成模型结合的成功范例。
*   **`One View Is Enough! Monocular Training for In-the-Wild Novel View Generation`**：**挑战性任务的简洁解决方案**。仅用单目视频训练即可实现野外场景的新视图生成，避免了多视图数据采集的昂贵成本和对精确相机参数的依赖，为3D内容创作提供了极具实用潜力的路径。

**3. 新兴研究方向与技术**

*   **推理驱动的生成与控制**：`UniGRPO`和`AgentRVOS`都强调将**逻辑推理**模块深度整合到视觉生成和分割任务中，标志着从“模式匹配”到“推理决策”的范式转变。
*   **动态世界建模与生成**：`WildWorld`数据集和`TETO`方法关注包含动作、状态和事件的动态序列，旨在构建可交互、可推演的虚拟世界模型，这为生成式AI在游戏、模拟和数字孪生中的应用铺平道路。
*   **大视觉语言模型的高效化**：`VISion On Request`提出的动态稀疏视觉-语言交互机制，是针对大视觉语言模型计算瓶颈的前沿解决方案，预示着下一代高效多模态模型的结构设计趋势。

**4. 推荐精读论文**

根据研究方向的普适性和影响力，建议优先阅读以下论文：

1.  **`Foveated Diffusion`**：**强烈推荐给所有生成模型研究者**。其效率优化思想具有广泛的启发性和迁移价值。
2.  **`CaP-X`**：**推荐给机器人学习、具身智能和AI智能体领域的研究者**。是了解该领域最新基准和方法的必读之作。
3.  **`One View Is Enough!`**：**推荐给3D视觉、神经渲染和生成模型的研究者**。其简洁而强大的方法可能成为单视图3D重建的新基线。
4.  **`VISion On Request`**：**推荐给大模型、多模态学习和模型压缩方向的研究者**。为解决VLM的计算效率问题提供了新颖的思路。

**总结**：今日的论文集合表明，计算机视觉研究正从独立的感知任务，快速向**高效、自适应、可推理**的**生成式智能体**和**动态世界模型**演进。研究重点兼顾了前沿探索（如具身推理）与实际部署（如效率优化），显示出该领域日益成熟的工程化与实用化倾向。

---

## Table of Contents

1. [CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation](#2603.22435v1)
2. [OccAny: Generalized Unconstrained Urban 3D Occupancy](#2603.23502v1)
3. [UniGRPO: Unified Policy Optimization for Reasoning-Driven Visual Generation](#2603.23500v1)
4. [DA-Flow: Degradation-Aware Optical Flow Estimation with Diffusion Models](#2603.23499v1)
5. [WildWorld: A Large-Scale Dataset for Dynamic World Modeling with Actions and Explicit State toward Generative ARPG](#2603.23497v1)
6. [VISion On Request: Enhanced VLLM efficiency with sparse, dynamically selected, vision-language interactions](#2603.23495v1)
7. [Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation](#2603.23491v1)
8. [AgentRVOS: Reasoning over Object Tracks for Zero-Shot Referring Video Object Segmentation](#2603.23489v1)
9. [One View Is Enough! Monocular Training for In-the-Wild Novel View Generation](#2603.23488v1)
10. [TETO: Tracking Events with Teacher Observation for Motion Estimation and Frame Interpolation](#2603.23487v1)

---

## Papers

<a id='2603.22435v1'></a>
## [CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation](https://arxiv.org/abs/2603.22435v1)

**Authors:** Max Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Fei-Fei Li, Guanya Shi, Jiajun Wu, Shankar Sastry, Yuke Zhu, Ken Goldberg, Linxi "Jim" Fan

**Published:** 2026-03-23

**Categories:** cs.RO, cs.AI

**Abstract:**

"Code-as-Policy" considers how executable code can complement data-intensive Vision-Language-Action (VLA) methods, yet their effectiveness as autonomous controllers for embodied manipulation remains underexplored. We present CaP-X, an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. At its core is CaP-Gym, an interactive environment in which agents control robots by synthesizing and executing programs that compose perception and control primitives. Building on this foundation, CaP-Bench evaluates frontier language and vision-language models across varying levels of abstraction, interaction, and perceptual grounding. Across 12 models, CaP-Bench reveals a consistent trend: performance improves with human-crafted abstractions but degrades as these priors are removed, exposing a dependence on designer scaffolding. At the same time, we observe that this gap can be mitigated through scaling agentic test-time computation--through multi-turn interaction, structured execution feedback, visual differencing, automatic skill synthesis, and ensembled reasoning--substantially improves robustness even when agents operate over low-level primitives. These findings allow us to derive CaP-Agent0, a training-free framework that recovers human-level reliability on several manipulation tasks in simulation and on real embodiments. We further introduce CaP-RL, showing reinforcement learning with verifiable rewards improves success rates and transfers from sim2real with minimal gap. Together, CaP-X provides a principled, open-access platform for advancing embodied coding agents.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对 **CaP-X** 这篇论文的分析如下：

### 1. 核心贡献总结
CaP-X 提出了一个用于评估与提升机器人操作“代码即策略”（Code-as-Policy）能力的开源框架。该研究通过 CaP-Bench 揭示了现有模型对人类设计抽象（Scaffolding）的过度依赖，并证明了通过多轮交互、执行反馈及自动技能合成等“测试时计算”（test-time computation）手段，可以有效弥补模型在低级基元操作上的鲁棒性缺陷。

### 2. 关键创新与方法论
*   **标准化评估体系（CaP-Gym & CaP-Bench）：** 构建了一个包含不同抽象层级、交互复杂度和感知要求的基准测试环境，填补了当前评估端到端 VLA 模型与代码生成智能体在操作任务上缺乏量化对比的空白。
*   **依赖性分析：** 系统性地揭示了代码生成模型性能与“人类预设抽象”之间的强耦合关系，指出了“模型越依赖高级接口，其泛化性可能越受限于预设的编程范式”这一痛点。
*   **测试时策略提升：** 引入了多轮交互、结构化反馈（Structured Execution Feedback）和视觉差异分析（Visual Differencing）。这种方法将单纯的“零样本代码生成”转变为“闭环执行与迭代优化”，显著提升了模型在低级基元控制下的可靠性。
*   **CaP-Agent0 与 CaP-RL：** 提供了一种无需训练的通用框架，并结合强化学习实现了可验证奖励（Verifiable Rewards），成功将仿真性能平滑迁移至真实机器人。

### 3. 对领域的潜在影响
*   **打破“黑盒”模型范式：** 计算机视觉领域正从单纯的感知任务转向具身智能。该研究强调的“代码生成”提供了一种可解释、可编程且模块化的控制方案，这为解决 VLA 模型在长程、复杂任务中缺乏透明度的挑战提供了有力补充。
*   **重新定义具身智能的路径：** 该论文可能引导社区从单纯追求模型参数规模（Scaling Laws for VLA），转向研究如何通过算法层面的测试时计算来提升小型模型或开源模型的执行鲁棒性。

### 4. 相关领域与应用前景
*   **工业自动化与柔性制造：** 无需针对新工件重写程序，智能体可根据视觉反馈自动合成操作代码。
*   **家庭服务机器人：** 在非结构化环境中，机器人需要处理环境的细微变化，CaP-X 的闭环反馈机制能有效提升机器人对突发状况的应对能力。
*   **多模态感知：** 将视觉信息与代码执行过程中的状态信息进行联合分析，对于提升视觉语言模型在动态场景下的定位（Grounding）与推理能力至关重要。

### 5. 可推断的局限性
*   **对基元库的依赖：** 尽管提出了提升鲁棒性的方法，但若底层机器人基元（Primitives）本身的控制精度不足，代码生成智能体即便逻辑正确也可能导致物理执行失败（Sim2Real 的底层偏差）。
*   **实时性挑战：** 论文中提到的“多轮交互”和“自动技能合成”虽然提升了成功率，但如果这些测试时计算逻辑过于复杂，可能会导致高延迟，从而限制其在要求极高实时响应的动态任务中的应用。
*   **任务定义的受限性：** 论文主要关注操纵（Manipulation）任务，对于更复杂、需要长期规划或大规模物体交互的导航等任务，现有的代码生成范式是否仍具扩展性尚待验证。

**专家总结：** 这篇论文的趣味性在于它从“代码生成”的角度切入了具身智能的核心瓶颈，证明了通过**引入推理反馈环（Agentic Feedback Loops）**，我们可以在不重新训练大型模型的情况下，极大地提升机器人对世界的操控能力。这对于计算机视觉开发者而言，是一个将“视觉感知”与“逻辑控制”有机结合的重要里程碑。

**Key Findings:**

- We present CaP-X, an open-access framework for systematically studying Code-as-Policy agents in robot manipulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.22435v1)
- [arXiv](https://arxiv.org/abs/2603.22435v1)

---

<a id='2603.23502v1'></a>
## [OccAny: Generalized Unconstrained Urban 3D Occupancy](https://arxiv.org/abs/2603.23502v1)

**Authors:** Anh-Quan Cao, Tuan-Hung Vu

**Published:** 2026-03-24

**Categories:** cs.CV

**Abstract:**

Relying on in-domain annotations and precise sensor-rig priors, existing 3D occupancy prediction methods are limited in both scalability and out-of-domain generalization. While recent visual geometry foundation models exhibit strong generalization capabilities, they were mainly designed for general purposes and lack one or more key ingredients required for urban occupancy prediction, namely metric prediction, geometry completion in cluttered scenes and adaptation to urban scenarios. We address this gap and present OccAny, the first unconstrained urban 3D occupancy model capable of operating on out-of-domain uncalibrated scenes to predict and complete metric occupancy coupled with segmentation features. OccAny is versatile and can predict occupancy from sequential, monocular, or surround-view images. Our contributions are three-fold: (i) we propose the first generalized 3D occupancy framework with (ii) Segmentation Forcing that improves occupancy quality while enabling mask-level prediction, and (iii) a Novel View Rendering pipeline that infers novel-view geometry to enable test-time view augmentation for geometry completion. Extensive experiments demonstrate that OccAny outperforms all visual geometry baselines on 3D occupancy prediction task, while remaining competitive with in-domain self-supervised methods across three input settings on two established urban occupancy prediction datasets. Our code is available at https://github.com/valeoai/OccAny .

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇论文《OccAny: Generalized Unconstrained Urban 3D Occupancy》的分析如下：

### 1. 核心贡献总结
OccAny 是首个针对城市场景设计的“无约束”（Unconstrained）3D 占用（Occupancy）预测框架，打破了传统方法对特定传感器校准和领域内标注数据的依赖。该模型实现了在未校准、跨域场景下的高精度度量级 3D 占用预测与几何补全，且能够灵活适配单目、多目及序列图像输入。

### 2. 关键创新与方法论
该论文的核心技术突破在于通过结合视觉几何基础模型与特定领域的优化手段，解决了通用模型在城市感知中的局限性：
*   **Segmentation Forcing（分割驱动）：** 该机制通过引入语义分割任务作为先验或辅助监督，强行约束模型理解场景的结构，显著提升了占用栅格的精细度，并赋予了模型 mask 级别的预测能力。
*   **Novel View Rendering Pipeline（新视角渲染流水线）：** 这是该模型的一大亮点。通过推断场景的新视角几何信息，利用测试时视点增强（Test-time view augmentation）来处理遮挡区域和进行几何补全，从而解决了单视角输入下的场景不完整问题。
*   **无约束架构设计：** 不再依赖严格的传感器内外参先验，使其能够处理来自不同来源、未经精心校准的城市图像数据。

### 3. 对领域的潜在影响
*   **从“专用”向“通用”的范式转移：** 传统的 3D 占用预测高度依赖特定数据集（如 nuScenes）的标注和特定的相机参数配置。OccAny 展示了利用大模型预训练能力实现“零样本/少样本”通用感知的可能性，这对于自动驾驶算法的规模化落地具有重要意义。
*   **几何与语义的深度融合：** 该研究证明了将语义分割特征融入几何占用预测不仅是有效的，而且是提升几何完备性的关键手段。
*   **降低部署成本：** 能够处理“未校准”场景，意味着该技术可以降低对昂贵硬件标定流程的依赖，提高系统在复杂多变环境下的鲁棒性。

### 4. 受益的相关领域与应用
*   **自动驾驶与机器人：** 能够直接应用于各种未经精确校准的量产车辆摄像头数据，增强感知系统在极端边缘案例（Edge Cases）下的环境理解能力。
*   **数字孪生与城市场景重建：** 利用其强大的几何补全能力，从低质量或受限的视觉输入中快速构建城市 3D 模型。
*   **AR/VR 环境感知：** 对于移动设备而言，该方法无需预设环境先验即可实现对周围空间的深度感知。

### 5. 可推断的局限性
*   **计算资源需求：** 尽管具备很强的泛化能力，但其 Novel View Rendering 流水线通常涉及大量的计算开销（类似于 NeRF 的优化过程），在车载实时处理环境（On-board real-time）下可能存在性能瓶颈。
*   **动态场景的鲁棒性：** 摘要中强调了几何补全，但在高度动态的城市交通环境中，多视角的一致性维护（尤其是在存在遮挡和运动物体时）依然是一个巨大挑战。
*   **基础模型偏差：** 作为基于视觉基础模型构建的架构，OccAny 的性能上限可能受限于预训练基座在城市特定视图（如俯视或广角畸变）下的先验知识。

---
**专家点评：**
OccAny 的趣味性在于它试图弥合“计算机视觉通用基础模型”与“自动驾驶精确几何感知”之间的鸿沟。在目前学术界趋向于“大一统”模型的背景下，该论文提供了一种务实的路径——即通过特定的几何推理任务（如新视角渲染）来补齐通用大模型在精细度量任务上的短板，是目前具身智能（Embodied AI）领域非常值得关注的研究方向。

**Key Findings:**

- Our contributions are three-fold: (i) we propose the first generalized 3D occupancy framework with (ii) Segmentation Forcing that improves occupancy quality while enabling mask-level prediction, and (iii) a Novel View Rendering pipeline that infers novel-view geometry to enable test-time view augmentation for geometry completion.
- Extensive experiments demonstrate that OccAny outperforms all visual geometry baselines on 3D occupancy prediction task, while remaining competitive with in-domain self-supervised methods across three input settings on two established urban occupancy prediction datasets.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.23502v1)
- [arXiv](https://arxiv.org/abs/2603.23502v1)

---

<a id='2603.23500v1'></a>
## [UniGRPO: Unified Policy Optimization for Reasoning-Driven Visual Generation](https://arxiv.org/abs/2603.23500v1)

**Authors:** Jie Liu, Zilyu Ye, Linxiao Yuan, Shenhan Zhu, Yu Gao, Jie Wu, Kunchang Li, Xionghui Wang, Xiaonan Nie, Weilin Huang, Wanli Ouyang

**Published:** 2026-03-24

**Categories:** cs.CV

**Abstract:**

Unified models capable of interleaved generation have emerged as a promising paradigm, with the community increasingly converging on autoregressive modeling for text and flow matching for image generation. To advance this direction, we propose a unified reinforcement learning framework tailored for interleaved generation. We validate our approach on its fundamental unit: a single round of reasoning-driven image generation, where the model first expands the user prompt through reasoning, followed by image synthesis. Formulating this multimodal generation process as a Markov Decision Process with sparse terminal rewards, we introduce UniGRPO to jointly optimize text and image generation policies using GRPO. Adopting a minimalist methodology to avoid over-design, we leverage established training recipes for both modalities by seamlessly integrating standard GRPO for reasoning and FlowGRPO for visual synthesis. To ensure scalability to multi-round interleaved generation, we introduce two critical modifications to the original FlowGRPO: (1) eliminating classifier-free guidance to maintain linear, unbranched rollouts, which is essential for scaling to complex scenarios involving multi-turn interactions and multi-condition generation (e.g., editing); and (2) replacing the standard latent KL penalty with an MSE penalty directly on the velocity fields, providing a more robust and direct regularization signal to mitigate reward hacking effectively. Our experiments demonstrate that this unified training recipe significantly enhances image generation quality through reasoning, providing a robust and scalable baseline for the future post-training of fully interleaved models.

**Analysis:**

### 1. 摘要翻译
统一的交错生成模型已成为极具前景的研究方向，目前社区正趋向于采用自回归建模处理文本、流匹配（Flow Matching）建模处理图像。为推动该领域发展，我们提出了 UniGRPO，这是一个专为交错生成设计的统一强化学习框架。我们在“推理驱动的图像生成”这一基本单元上验证了该方法，即模型首先通过推理扩展用户提示词，随后进行图像合成。通过将多模态生成过程建模为具有稀疏终端奖励的马尔可夫决策过程（MDP），我们利用 GRPO 联合优化文本与图像生成策略。为避免过度设计，我们采用极简方法，将标准 GRPO（用于推理）与 FlowGRPO（用于图像合成）无缝集成。为确保向多轮交错生成的可扩展性，我们引入了两项关键改进：(1) 消除分类器自由引导（CFG）以保持线性、无分支的推演过程，这对处理复杂的多轮交互和多条件生成场景至关重要；(2) 用直接作用于速度场的 MSE 惩罚代替标准隐空间 KL 惩罚，提供更稳健、直接的正则化信号以有效抑制奖励欺骗。实验表明，该统一训练方案显著提升了推理驱动下的图像生成质量，为未来完全交错模型的后训练提供了稳健且可扩展的基线。

### 2. 方法动机分析
- **驱动力**：旨在构建一个能统一处理“思考（推理）+生成（图像）”的单一模型，打破目前两个模态在RL优化上的割裂。
- **痛点**：
    - **CFG的局限性**：CFG在多轮交互和多条件生成中会导致计算复杂度和上下文分支激增，难以进行有效的梯度估计。
    - **奖励欺骗**：现有的基于KL散度的正则化在不同时间步上的惩罚力度不均，导致模型容易通过“走捷径”来最大化奖励，牺牲图像质量。
- **研究假设**：通过将推理与生成合并为单一 MDP，并配合稳健的正则化手段，可以引导模型学习出更具针对性的思维链（CoT），从而显著提升图像生成质量。

### 3. 方法设计详解
- **核心逻辑**：UniGRPO 将任务建模为 $S, A, P, R$ 的 MDP。
    - **状态空间**：包含当前提示词、历史推理 tokens、图像隐变量及流时间。
    - **动作空间**：文本为 token 预测，图像为去噪隐变量的迭代更新。
- **技术细节**：
    - **无CFG训练**：移除推理时的 CFG，强制模型将对齐能力（提示词依从性）通过 RL 损失直接内化到策略权重中，保证计算图是线性的。
    - **基于速度的正则化 (RatioNorm)**：放弃 latent KL 惩罚（因为它在不同噪声水平下加权不均），改用对速度场（velocity fields）直接进行 MSE 惩罚。该惩罚无时间步加权，约束模型生成的向量场在全过程都贴近预训练的参考模型，彻底杜绝了时间依赖带来的漏洞。
    - **联合优化**：利用 GRPO 的 group-relative advantage，对文本链和图像去噪步骤进行端到端的联合优化，让“推理”服务于“生成”。

### 4. 方法对比分析
- **本质区别**：与以往分离优化不同（如 TextGRPO 与 FlowGRPO 分开调优），UniGRPO 实现了完全的联合端到端优化。
- **创新贡献**：引入了 RatioNorm 机制改进了 FlowGRPO，并首次在推理驱动的视觉生成中去除了训练时的 CFG，极大地简化了多轮交互的训练拓扑。
- **适用场景**：特别适用于需要复杂思维链辅助生成、多步骤提示词重写及多条件图像编辑任务的统一多模态模型。

### 5. 实验分析
- **验证方法**：在文本对齐（TA）和 GenEval 基准上对比了 ReFL、FPO、TextGRPO 等主流基线。
- **关键结果**：UniGRPO 在 TA (0.8381) 和 GenEval (0.90) 上均取得 SOTA，且证明了联合优化优于单模态优化。
- **优势/局限**：优势在于训练过程极其稳健，能够生成高质量、符合逻辑的思维链；局限在于目前仍依赖最终图像的终端奖励（sparse reward），缺乏对中间推理步骤的精细化监督。

### 6. 实用指南
- **开源/复现**：代码与模型参考 ByteDance Seed 团队相关工作（如 Bagel 架构）。
- **关键细节**：正则化系数（MSE Loss Weight: 1.5e-5）和梯度剪裁范围（Loss Clip Range: 1e-6）是保证 RL 训练不坍缩的核心。
- **迁移性**：该框架的“无CFG”和“速度场MSE正则化”方案可直接迁移至任何基于 Flow Matching 的生成模型中。

### 7. 总结
- **核心思想**：通过线性化推演与速度场正则化，统一文本推理与视觉生成的端到端 RL。
- **速记版 pipeline**：
    1. 生成多组推理链（思维链）；
    2. 基于思维链进行流匹配图像去噪；
    3. 计算终端图像奖励；
    4. 执行无CFG下的GRPO梯度更新；
    5. MSE 约束速度场防止奖励欺骗。

**Key Findings:**

- To advance this direction, we propose a unified reinforcement learning framework tailored for interleaved generation.
- We validate our approach on its fundamental unit: a single round of reasoning-driven image generation, where the model first expands the user prompt through reasoning, followed by image synthesis.
- Formulating this multimodal generation process as a Markov Decision Process with sparse terminal rewards, we introduce UniGRPO to jointly optimize text and image generation policies using GRPO.
- To ensure scalability to multi-round interleaved generation, we introduce two critical modifications to the original FlowGRPO: (1) eliminating classifier-free guidance to maintain linear, unbranched rollouts, which is essential for scaling to complex scenarios involving multi-turn interactions and multi-condition generation (e.g., editing); and (2) replacing the standard latent KL penalty with an MSE penalty directly on the velocity fields, providing a more robust and direct regularization signal to mitigate reward hacking effectively.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.23500v1)
- [arXiv](https://arxiv.org/abs/2603.23500v1)

---

<a id='2603.23499v1'></a>
## [DA-Flow: Degradation-Aware Optical Flow Estimation with Diffusion Models](https://arxiv.org/abs/2603.23499v1)

**Authors:** Jaewon Min, Jaeeun Lee, Yeji Choi, Paul Hyunbin Cho, Jin Hyeon Kim, Tae-Young Lee, Jongsik Ahn, Hwayeong Lee, Seonghyun Park, Seungryong Kim

**Published:** 2026-03-24

**Categories:** cs.CV

**Abstract:**

Optical flow models trained on high-quality data often degrade severely when confronted with real-world corruptions such as blur, noise, and compression artifacts. To overcome this limitation, we formulate Degradation-Aware Optical Flow, a new task targeting accurate dense correspondence estimation from real-world corrupted videos. Our key insight is that the intermediate representations of image restoration diffusion models are inherently corruption-aware but lack temporal awareness. To address this limitation, we lift the model to attend across adjacent frames via full spatio-temporal attention, and empirically demonstrate that the resulting features exhibit zero-shot correspondence capabilities. Based on this finding, we present DA-Flow, a hybrid architecture that fuses these diffusion features with convolutional features within an iterative refinement framework. DA-Flow substantially outperforms existing optical flow methods under severe degradation across multiple benchmarks.

**Analysis:**

以下是对论文《DA-Flow: Degradation-Aware Optical Flow Estimation with Diffusion Models》的深度分析：

### 1. 摘要翻译
光学流模型在高质量数据上训练，但在遇到模糊、噪声和压缩伪影等真实世界损坏时性能会显著下降。为了克服这一局限性，我们提出了**退化感知光流（Degradation-Aware Optical Flow）**，这是一项针对从真实世界受损视频中进行精确稠密对应估计的新任务。我们的核心洞察是：图像恢复扩散模型的中间表示天然具有退化感知能力，但缺乏时间上的感知。为了解决这一不足，我们通过全时空注意力机制（full spatio-temporal attention）提升了模型对相邻帧的处理能力，并实证表明所得特征展现出了零样本对应能力。基于此发现，我们提出了 DA-Flow，这是一种混合架构，在迭代细化框架内融合了扩散特征与卷积特征。DA-Flow 在多种基准测试下的严重退化场景中大幅超越了现有的光流方法。

### 2. 方法动机分析
- **驱动力**：现有的光流模型在处理“干净”视频时表现卓越，但在面临真实场景中普遍存在的低质量退化（如噪声、压缩、模糊）时，无法提取稳定的特征进行匹配。
- **痛点**：简单地进行数据增强无法解决退化带来的特征空间分布偏移；且现有的视频恢复扩散模型多采用3D卷积或早期时序压缩，这破坏了对光流估计至关重要的单帧空间结构。
- **假设**：预训练的图像恢复扩散模型（Image Restoration Diffusion Models）的中间特征不仅包含几何结构，还编码了恢复“干净”图像所需的退化模式知识，如果能通过时空注意力将其“激活”并提取，即可作为鲁棒的光流特征。

### 3. 方法设计详解
- **流程总结**：
    1. **提升（Lifting）**：将预训练的图像恢复模型（DiT4SR）通过在各层注入全时空注意力机制，转换为视频处理模型，从而在保持独立空间结构的同时实现跨帧信息交互。
    2. **特征提取**：在迭代去噪过程中，从提升后的模型中提取Query和Key特征。
    3. **特征融合**：将提取的扩散特征通过DPT（Dense Prediction Transformer）头进行上采样，并与传统的CNN编码器（RAFT架构）提取的特征拼接。
    4. **迭代更新**：利用混合特征构建代价体（Cost Volume），通过原有的迭代细化机制（Update Operator）输出最终光流。
- **核心逻辑**：该方法本质上是将预训练的图像生成模型作为“特征骨干”，利用其丰富的退化感知先验来增强经典光流架构的鲁棒性。

### 4. 方法对比分析
- **本质区别**：与传统直接在退化图像上训练的模型不同，DA-Flow 利用了**生成式先验**，通过扩散模型的中间表示，在噪声中“重构”出几何对应关系，而非硬编码处理噪声。
- **创新贡献**：提出了一种将图像级恢复模型“提升”为视频处理模型的通用架构设计（通过Reshape + Spatio-temporal Attention），避免了视频扩散模型常见的时序塌陷问题。

### 5. 实验分析
- **验证方法**：在Sintel、Spring和TartanAir等基准测试上加入 Realistic Degradation Pipeline 引入合成退化。
- **关键结论**：DA-Flow 在所有评估指标（EPE及异常值率）上均显著优于现有模型（RAFT、SEA-RAFT、FlowSeek）。
- **主要优势**：在极端退化场景下表现极其稳定，能够有效恢复出物体边界和细粒度结构。
- **主要局限**：由于需要在推理阶段运行多次去噪步骤，导致计算延迟比传统单次前向传导的光流模型更高。

### 6. 实用指南
- **开源情况**：已提供项目主页（https://cvlab-kaist.github.io/DA-Flow）。
- **迁移建议**：该“提升”架构（DiT Lifting）具有高度通用性，可以迁移到其他依赖于几何结构但受限于输入质量的任务，如单目深度估计或视频分割。
- **实现注意**：在提取特征时，不要在单一去噪步提取，需分析不同Denosing Step的特征稳定性，选择最佳层。

### 7. 总结
- **核心思想**：利用图像恢复扩散模型的先验知识，通过时空提升实现退化场景下的鲁棒光流估计。
- **速记版pipeline**：
    1. 修改扩散模型架构，加入全时空注意力机制。
    2. 提取扩散过程中的中间特征。
    3. 将扩散特征与传统CNN视觉特征拼接。
    4. 放入光流迭代网络完成预测。

**Key Findings:**

- To overcome this limitation, we formulate Degradation-Aware Optical Flow, a new task targeting accurate dense correspondence estimation from real-world corrupted videos.
- Based on this finding, we present DA-Flow, a hybrid architecture that fuses these diffusion features with convolutional features within an iterative refinement framework.
- DA-Flow substantially outperforms existing optical flow methods under severe degradation across multiple benchmarks.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.23499v1)
- [arXiv](https://arxiv.org/abs/2603.23499v1)

---

<a id='2603.23497v1'></a>
## [WildWorld: A Large-Scale Dataset for Dynamic World Modeling with Actions and Explicit State toward Generative ARPG](https://arxiv.org/abs/2603.23497v1)

**Authors:** Zhen Li, Zian Meng, Shuwei Shi, Wenshuo Peng, Yuwei Wu, Bo Zheng, Chuanhao Li, Kaipeng Zhang

**Published:** 2026-03-24

**Categories:** cs.CV

**Abstract:**

Dynamical systems theory and reinforcement learning view world evolution as latent-state dynamics driven by actions, with visual observations providing partial information about the state. Recent video world models attempt to learn this action-conditioned dynamics from data. However, existing datasets rarely match the requirement: they typically lack diverse and semantically meaningful action spaces, and actions are directly tied to visual observations rather than mediated by underlying states. As a result, actions are often entangled with pixel-level changes, making it difficult for models to learn structured world dynamics and maintain consistent evolution over long horizons. In this paper, we propose WildWorld, a large-scale action-conditioned world modeling dataset with explicit state annotations, automatically collected from a photorealistic AAA action role-playing game (Monster Hunter: Wilds). WildWorld contains over 108 million frames and features more than 450 actions, including movement, attacks, and skill casting, together with synchronized per-frame annotations of character skeletons, world states, camera poses, and depth maps. We further derive WildBench to evaluate models through Action Following and State Alignment. Extensive experiments reveal persistent challenges in modeling semantically rich actions and maintaining long-horizon state consistency, highlighting the need for state-aware video generation. The project page is https://shandaai.github.io/wildworld-project/.

**Analysis:**

## 1. 摘要翻译
本研究提出了WildWorld，一个从AAA动作角色扮演游戏（《怪物猎人：荒野》）中自动采集的、包含显式状态标注的大规模动作条件世界建模数据集。WildWorld拥有超过1.08亿帧数据，涵盖450多种动作，并提供帧级同步的角色骨架、世界状态、相机位姿及深度图。为评估模型性能，我们构建了WildBench基准，包含“动作遵循”和“状态对齐”两个指标。实验揭示了当前模型在建模语义丰富动作和保持长时状态一致性方面存在显著挑战，强调了状态感知视频生成的重要性。

## 2. 方法动机分析
*   **驱动力**：现有的视频世界模型大多基于纯视觉观察学习动态，导致动作与像素级变化纠缠，难以解耦真正的状态转换，进而导致长时预测失效。
*   **现有痛点**：现有数据集缺乏明确的、语义化的状态标注，且动作往往简单（如基础位移），难以支撑复杂的、由底层状态驱动的环境模拟。
*   **研究假设**：通过显式的、语义丰富的状态作为中间表示，可以辅助解耦动作与视觉观察，从而提升长时状态一致性和可预测性。

## 3. 方法设计详解
*   **数据采集 pipeline**：
    1.  **多源同步录制**：通过Obs Studio和自定义Reshade Shader将RGB、深度、相机位姿、骨架数据与游戏引擎内部的动作ID、状态变量（HP、位置、动画帧等）精确同步至JSON文件。
    2.  **多维过滤**：引入时长、时间连续性（剔除卡顿/过场）、亮度、相机/字符遮挡等过滤策略，保证样本质量。
    3.  **层级标注**：将样本按Action ID分段，每段利用Qwen3-VL-235B生成动作级描述，再汇总生成样本级概括。
*   **StateCtrl 模型结构**：
    *   **状态嵌入层**：离散状态通过Embedding，连续状态通过MLP映射，采用层级建模（实体级+全局上下文）。
    *   **状态转换建模**：使用Transformer建模状态间关系。
    *   **注入机制**：将状态Embedding作为DiT（Diffusion Transformer）的中间条件注入，引导生成。
    *   **监督辅助**：通过解码损失（保持状态）和预测损失（预测下帧状态）约束状态空间的结构化。

## 4. 方法对比分析
*   **本质区别**：从“像素到像素”的端到端学习，转向“状态感知”的显式建模，将状态作为交互与像素生成的中间桥梁。
*   **创新贡献**：首次在AAA游戏环境下构建了包含亿级帧、多模态、全状态监督的大规模数据集及配套的定量基准（WildBench）。
*   **适用场景**：适用于需要长时一致性、具备复杂状态交互（如战斗系统）的交互式环境模拟。

## 5. 实验分析（精简版）
*   **关键结论**：实验证明基于状态注入的StateCtrl在动作遵循和状态对齐指标上显著优于基线模型；同时指出，纯视频质量指标（MS/DD）在当前数据集下趋于饱和，传统的视觉指标已无法衡量交互模型的真实能力。
*   **优势**：引入了量化的“状态对齐”指标，为交互模型提供了更细粒度的评估依据。
*   **局限**：自回归预测机制在长时生成中存在累积误差，且复杂状态的加入在提升控制力的同时，可能牺牲部分视觉质量。

## 6. 实用指南
*   **开源情况**：已开源数据集与基准测试代码（https://github.com/ShandaAI/WildWorld）。
*   **实现细节**：建议使用DiT架构作为骨干；在训练中，必须采用多任务学习（辅助预测损失）以确保隐空间保留状态信息；采样帧率维持在16 FPS左右较为均衡。
*   **迁移可能**：该框架可直接迁移至其他动作丰富的游戏引擎（如Unity/UE开发的其他动作游戏）或具备强状态约束的仿真机器人模拟任务。

## 7. 总结
*   **核心思想**：利用显式状态标注辅助动作条件下的世界模型构建与评估。
*   **速记版pipeline**：
    1. 游戏引擎状态流采集与多源同步；
    2. 多维度数据过滤与分段层级标注；
    3. 状态结构化建模与DiT注入；
    4. 动作遵循与状态对齐指标评估。

**Key Findings:**

- In this paper, we propose WildWorld, a large-scale action-conditioned world modeling dataset with explicit state annotations, automatically collected from a photorealistic AAA action role-playing game (Monster Hunter: Wilds).

**Links:**

- [PDF](https://arxiv.org/pdf/2603.23497v1)
- [arXiv](https://arxiv.org/abs/2603.23497v1)

---

<a id='2603.23495v1'></a>
## [VISion On Request: Enhanced VLLM efficiency with sparse, dynamically selected, vision-language interactions](https://arxiv.org/abs/2603.23495v1)

**Authors:** Adrian Bulat, Alberto Baldrati, Ioannis Maniadis Metaxas, Yassine Ouali, Georgios Tzimiropoulos

**Published:** 2026-03-24

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Existing approaches for improving the efficiency of Large Vision-Language Models (LVLMs) are largely based on the concept of visual token reduction. This approach, however, creates an information bottleneck that impairs performance, especially on challenging tasks that require fine-grained understanding and reasoning. In this work, we challenge this paradigm by introducing VISion On Request (VISOR), a method that reduces inference cost without discarding visual information. Instead of compressing the image, VISOR improves efficiency by sparsifying the interaction between image and text tokens. Specifically, the language model attends to the full set of high-resolution visual tokens through a small, strategically placed set of attention layers: general visual context is provided by efficient cross-attention between text-image, while a few well-placed and dynamically selected self-attention layers refine the visual representations themselves, enabling complex, high-resolution reasoning when needed. Based on this principle, we first train a single universal network on a range of computational budgets by varying the number of self-attention layers, and then introduce a lightweight policy mechanism that dynamically allocates visual computation based on per-sample complexity. Extensive experiments show that VISOR drastically reduces computational cost while matching or exceeding state-of-the-art results across a diverse suite of benchmarks, and excels in challenging tasks that require detailed visual understanding.

**Analysis:**

这是一份关于《VISion On Request: Enhanced VLLM efficiency with sparse, dynamically selected, vision-language interactions》的深度技术分析：

### 1. 摘要翻译
现有的大型视觉语言模型（LVLMs）效率优化方法多基于视觉token减少技术，这往往导致信息瓶颈，损害复杂推理任务的性能。本文提出了“按需视觉”（VISOR）方法。VISOR不压缩图像，而是通过稀疏化图像与文本token之间的交互来降低推理成本。模型利用一组小型、策略性放置的注意力层来访问全分辨率视觉特征，并引入轻量级路由机制，根据样本复杂度动态分配视觉计算。实验表明，VISOR在大幅降低计算成本的同时，在复杂视觉理解任务上匹配或超过了现有技术水平。

### 2. 方法动机分析
*   **驱动力**：打破现有“视觉token压缩导致信息瓶颈”的局限，在保持全分辨率视觉信息的前提下，通过减少计算量来实现效率提升。
*   **现有痛点**：现有的token pruning/merging方法在处理需要精细视觉（如文档理解、图表问答）的任务时，会永久性丢失细节，造成不可逆的信息损失。
*   **核心假设**：LVLM中视觉token与文本token的交互是高度稀疏的，且这种稀疏性与任务复杂度密切相关。即：不需要每层都进行密集的视觉交互，部分层仅需处理文本或执行简单的视觉查询。

### 3. 方法设计详解
*   **流程总结（Pipeline）**：
    1.  **解耦交互**：将传统的Transformer层解构，大部分层只处理文本，仅在特定的层（LCA集合）进行Cross-Attention（提供视觉上下文），在更少的层（LSA集合）进行Self-Attention（更新视觉特征）。
    2.  **Universal Network训练**：预训练一个支持多种计算路径（配置）的单模型，通过在不同层级执行或跳过自注意力层来动态调整预算。
    3.  **动态路由（Adaptive Inference）**：利用一个轻量级策略网络（基于路由token的MLP），在推理阶段根据当前样本的复杂度实时决定需要执行哪些层。
*   **算法逻辑**：核心公式在于对第$l$层的处理逻辑：若$l \in L_{SA}$，进行全自注意力更新（代价高）；若$l \in L_{CA}$，仅执行Cross-Attention（代价低，视觉token不变）；其余层则跳过视觉计算。
*   **关键组件**：采用了1D深度可分离卷积实现条件位置编码，在保持空间信息的同时，避免了旋转位置编码在大规模下收敛慢的问题。

### 4. 方法对比分析
*   **本质区别**：传统方法是“空间/token缩减”（砍掉一部分视觉信息），VISOR是“深度/交互缩减”（保留全量视觉信息，但减少与它的计算交互次数）。
*   **创新贡献**：提出了一种正交于token压缩的效率优化路径，证明了任务复杂度决定了视觉特征的更新需求。
*   **适用场景**：极高分辨率图像的复杂推理任务，尤其是那些对细节丢失极其敏感的场景。

### 5. 实验分析
*   **关键结论**：在保持LLaVA-OV全分辨率输入的情况下，VISOR实现了高达8.6×-18×的FLOPs节省。
*   **优势**：在“Hard”任务（DocVQA, ChartQA）上显著优于现有的裁剪方法，不存在性能退化。
*   **局限**：需要针对不同的模型 Backbone 重新进行 universal 训练，且路由策略的训练依赖于伪标签过程，有一定稳定性开销。

### 6. 实用指南
*   **实现细节**：建议将Cross-Attention层均匀分布在模型中，自注意力层则选择性地置于重要逻辑节点（如每1/3层）。
*   **迁移与应用**：该方法完全可迁移至任意基于Transformer的VLLM。迁移核心在于：无需修改原始模型权重，只需增加并微调轻量级的CA/SA适配器层，并训练一个简单的路由MLP即可。
*   **数据预处理**：必须保留原始高分辨率图像，因为VISOR的初衷正是利用这些细节而非压缩它们。

### 7. 总结
*   **核心思想**：通过按需动态分配视觉交互深度，在保留全量细节的同时实现推理加速。
*   **速记版Pipeline**：
    1. 训练一个包含多层选择空间的全能模型。
    2. 对每个输入，通过路由标记预判计算需求。
    3. 仅在必要时执行高代价的视觉更新，大部分层仅进行轻量文本查询。

**Key Findings:**

- Extensive experiments show that VISOR drastically reduces computational cost while matching or exceeding state-of-the-art results across a diverse suite of benchmarks, and excels in challenging tasks that require detailed visual understanding.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.23495v1)
- [arXiv](https://arxiv.org/abs/2603.23495v1)

---

<a id='2603.23491v1'></a>
## [Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation](https://arxiv.org/abs/2603.23491v1)

**Authors:** Brian Chao, Lior Yariv, Howard Xiao, Gordon Wetzstein

**Published:** 2026-03-24

**Categories:** cs.CV

**Abstract:**

Diffusion and flow matching models have unlocked unprecedented capabilities for creative content creation, such as interactive image and streaming video generation. The growing demand for higher resolutions, frame rates, and context lengths, however, makes efficient generation increasingly challenging, as computational complexity grows quadratically with the number of generated tokens. Our work seeks to optimize the efficiency of the generation process in settings where the user's gaze location is known or can be estimated, for example, by using eye tracking. In these settings, we leverage the eccentricity-dependent acuity of human vision: while a user perceives very high-resolution visual information in a small region around their gaze location (the foveal region), the ability to resolve detail quickly degrades in the periphery of the visual field. Our approach starts with a mask modeling the foveated resolution to allocate tokens non-uniformly, assigning higher token density to foveal regions and lower density to peripheral regions. An image or video is generated in a mixed-resolution token setting, yielding results perceptually indistinguishable from full-resolution generation, while drastically reducing the token count and generation time. To this end, we develop a principled mechanism for constructing mixed-resolution tokens directly from high-resolution data, allowing a foveated diffusion model to be post-trained from an existing base model while maintaining content consistency across resolutions. We validate our approach through extensive analysis and a carefully designed user study, demonstrating the efficacy of foveation as a practical and scalable axis for efficient generation.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇题为《Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation》的论文分析如下：

### 1. 主要贡献总结
该论文提出了一种基于**注视点（Foveated）自适应机制**的扩散模型生成方法，旨在解决高分辨率图像和视频生成中计算复杂度过高的问题。通过利用人类视觉系统在注视点区域高分辨率、周边区域低分辨率的感知特性，该方法实现了非均匀的Token分配，在保持感知质量的前提下显著降低了生成过程中的计算成本。

### 2. 关键创新与方法论
*   **空间自适应Token建模**：核心创新在于引入了一种“混合分辨率”Token生成架构。模型不再对整个画面进行同等强度的采样，而是根据注视点位置生成空间分布不均的Token密度图。
*   **注视点感知映射**：该方法利用人类视网膜的偏心率（eccentricity）视敏度下降规律，动态调整像素空间的Token采样率，从而在感知质量不下降的情况下大幅压缩计算量。
*   **高效的后训练（Post-training）范式**：该工作设计了一套机制，允许在不重头训练的情况下，利用现有的预训练扩散模型进行微调，从而在保持语义一致性的同时实现“注视点驱动”的生成。

### 3. 对该领域的潜在影响
*   **计算效率的范式转移**：目前生成式AI的性能瓶颈主要在于计算开销随分辨率和帧率呈平方级增长。该论文引入的“感知效率”概念，将计算预算从单纯的全局计算转移到对人眼真正重要的区域，是实现实时超高清流媒体生成的重要路径。
*   **生成式AI与人机交互（HCI）的深度融合**：该研究展示了计算视觉心理学如何反哺生成式模型，通过引入眼动追踪反馈，使模型变得更加“智能且节约”，为下一代交互式生成内容提供了理论基石。

### 4. 受益的相关领域与应用
*   **XR/VR/AR（扩展现实）**：在虚拟现实头显中，渲染负担是核心痛点。该技术可以直接整合进头戴显示设备的注视点追踪系统，实现极高画质下的实时生成。
*   **实时流媒体与云游戏**：在带宽受限或实时渲染延迟敏感的场景中，该技术能显著降低服务器端的计算压力和端到端的传输延迟。
*   **交互式创意工具**：对于数字绘画、实时视频修补等需要交互响应的场景，用户可以在交互过程中即时获得高精度的局部细节，无需等待全图生成。

### 5. 可推断的潜在限制
*   **注视点追踪依赖**：该方法高度依赖准确的眼动追踪（Eye-tracking）硬件或鲁棒的注视点估计模型。如果追踪不准确或延迟较大，用户会直接察觉到周边区域的模糊甚至产生视觉伪影（如闪烁）。
*   **动态场景的挑战**：如果注视点移动非常频繁或快速（扫视），非均匀Token的重构可能导致画面在时序上出现不一致或出现“滞后模糊”。
*   **语义完整性问题**：虽然视觉上不可感知，但在生成过程中，低分辨率周边区域是否可能出现语义逻辑错误（即“幻觉”现象），或者在视线移动后，从周边进入视野的区域能否快速补充高频细节而不产生明显过渡痕迹，仍需在复杂动态环境下验证。

**总结评价：** 这篇论文的趣味性在于它巧妙地将**人类生物视觉的演化特性**转化为一种**算法优化策略**。在算力愈发紧张的AI时代，这种“按需计算”的思想比单纯堆砌硬件更具有长远的工程价值，是连接感知科学与生成式AI的一个高水准研究范例。

**Key Findings:**

- Our approach starts with a mask modeling the foveated resolution to allocate tokens non-uniformly, assigning higher token density to foveal regions and lower density to peripheral regions.
- To this end, we develop a principled mechanism for constructing mixed-resolution tokens directly from high-resolution data, allowing a foveated diffusion model to be post-trained from an existing base model while maintaining content consistency across resolutions.
- We validate our approach through extensive analysis and a carefully designed user study, demonstrating the efficacy of foveation as a practical and scalable axis for efficient generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.23491v1)
- [arXiv](https://arxiv.org/abs/2603.23491v1)

---

<a id='2603.23489v1'></a>
## [AgentRVOS: Reasoning over Object Tracks for Zero-Shot Referring Video Object Segmentation](https://arxiv.org/abs/2603.23489v1)

**Authors:** Woojeong Jin, Jaeho Lee, Heeseong Shin, Seungho Jang, Junhwan Heo, Seungryong Kim

**Published:** 2026-03-24

**Categories:** cs.CV

**Abstract:**

Referring Video Object Segmentation (RVOS) aims to segment a target object throughout a video given a natural language query. Training-free methods for this task follow a common pipeline: a MLLM selects keyframes, grounds the referred object within those frames, and a video segmentation model propagates the results. While intuitive, this design asks the MLLM to make temporal decisions before any object-level evidence is available, limiting both reasoning quality and spatio-temporal coverage. To overcome this, we propose AgentRVOS, a training-free agentic pipeline built on the complementary strengths of SAM3 and a MLLM. Given a concept derived from the query, SAM3 provides reliable perception over the full spatio-temporal extent through generated mask tracks. The MLLM then identifies the target through query-grounded reasoning over this object-level evidence, iteratively pruning guided by SAM3's temporal existence information. Extensive experiments show that AgentRVOS achieves state-of-the-art performance among training-free methods across multiple benchmarks, with consistent results across diverse MLLM backbones. Our project page is available at: https://cvlab-kaist.github.io/AgentRVOS/.

**Analysis:**

作为计算机视觉领域的专家，我对这篇关于 **AgentRVOS** 的论文分析如下：

### 1. 论文核心贡献总结
AgentRVOS 提出了一种创新的“无需训练”（Training-free）的代理式流水线（Agentic Pipeline），旨在解决指代视频目标分割（RVOS）中时空推理不足的问题。该方法通过将感知任务（由 SAM3 完成）与推理任务（由 MLLM 完成）解耦并协同，实现了从“先决策后感知”到“基于时空轨迹证据进行推理”的范式转变，从而显著提升了在复杂视频场景下的分割精度。

### 2. 关键创新与方法论
*   **范式转换（Evidence-First Reasoning）：** 传统方法要求 MLLM 在缺乏足够信息的情况下直接进行关键帧决策，而 AgentRVOS 首先利用 SAM3 对视频进行全时空的 Mask 轨迹生成（Object-level Evidence），确保了“先有证据，后做判断”。
*   **协同机制：** 该方法构建了一个闭环的交互机制。SAM3 提供稠密的时空掩码轨迹作为上下文，MLLM 在此基础上进行查询引导的推理与筛选。
*   **迭代修剪策略：** 引入了基于 SAM3 时空存在性（Temporal Existence）信息的迭代修剪过程，使 MLLM 能够准确剔除不相关的轨迹，从而精确定位目标对象。

### 3. 对计算机视觉领域的潜在影响
*   **打破了对大规模数据集的依赖：** 证明了无需特定任务微调（Training-free），仅通过多模态大模型（MLLM）与强力视觉分割模型（如 SAM 系列）的协同，即可达到 SOTA 性能，这为资源受限场景下的视频理解提供了新路径。
*   **代理式视觉（Agentic Vision）的实证：** 论文展示了通过将 MLLM 作为“大脑”来调度视觉工具（SAM3）进行复杂推理的潜力，预示了未来视觉任务将更多地转向由 LLM/MLLM 驱动的模块化协作架构。

### 4. 相关领域与受益应用
*   **视频分析与监控：** 对于需要精确追踪特定描述对象的智能安防系统。
*   **人机交互（HCI）：** 能够让机器人通过自然语言指令在复杂的动态环境（如家庭或仓库）中识别并跟踪目标。
*   **视频编辑工具：** 自动化的视频剪辑与特效处理，用户只需输入语言描述即可自动提取对应物体。
*   **自动驾驶场景理解：** 在复杂动态场景中对交通参与者或特定障碍物进行实时指代追踪。

### 5. 可推断的局限性
*   **计算开销：** 由于涉及全视频尺度的 SAM3 轨迹生成和 MLLM 的多轮迭代推理，推理延迟可能较高，难以直接实现高帧率的实时处理。
*   **对基座模型能力的依赖：** 性能表现高度依赖于所使用的 MLLM 的推理能力（如指令遵循、跨模态对齐能力）以及 SAM3 的分割泛化能力。若视频中出现严重的遮挡、形变或模糊，SAM3 产生的轨迹质量将成为性能瓶颈。
*   **长视频处理能力：** 虽然通过迭代修剪提升了效率，但对于极长视频的上下文窗口限制（Context Window Limit）及跨长时序的一致性仍可能存在潜在挑战。

**总结：** AgentRVOS 是一项极具前瞻性的工作，它巧妙地利用了“感知（SAM3）+ 推理（MLLM）”的二元协同，为视频理解领域的 Training-free 方向提供了一个高质量的框架范例，体现了当前从“端到端模型”向“智能代理架构”演进的技术趋势。

**Key Findings:**

- To overcome this, we propose AgentRVOS, a training-free agentic pipeline built on the complementary strengths of SAM3 and a MLLM.
- Extensive experiments show that AgentRVOS achieves state-of-the-art performance among training-free methods across multiple benchmarks, with consistent results across diverse MLLM backbones.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.23489v1)
- [arXiv](https://arxiv.org/abs/2603.23489v1)

---

<a id='2603.23488v1'></a>
## [One View Is Enough! Monocular Training for In-the-Wild Novel View Generation](https://arxiv.org/abs/2603.23488v1)

**Authors:** Adrien Ramanana Rahary, Nicolas Dufour, Patrick Perez, David Picard

**Published:** 2026-03-24

**Categories:** cs.CV

**Abstract:**

Monocular novel-view synthesis has long required multi-view image pairs for supervision, limiting training data scale and diversity. We argue it is not necessary: one view is enough. We present OVIE, trained entirely on unpaired internet images. We leverage a monocular depth estimator as a geometric scaffold at training time: we lift a source image into 3D, apply a sampled camera transformation, and project to obtain a pseudo-target view. To handle disocclusions, we introduce a masked training formulation that restricts geometric, perceptual, and textural losses to valid regions, enabling training on 30 million uncurated images. At inference, OVIE is geometry-free, requiring no depth estimator or 3D representation. Trained exclusively on in-the-wild images, OVIE outperforms prior methods in a zero-shot setting, while being 600x faster than the second-best baseline. Code and models are publicly available at https://github.com/AdrienRR/ovie.

**Analysis:**

作为计算机视觉领域的专家，我对这篇论文《One View Is Enough! Monocular Training for In-the-Wild Novel View Generation》的分析如下：

### 1. 论文核心贡献总结
该论文提出了一种名为 **OVIE** 的新型视角合成框架，打破了传统方法对多视角图像对（multi-view pairs）的依赖，实现了仅利用非配对的互联网图像进行训练。其核心贡献在于证明了通过单目深度估计作为几何先验，结合掩码训练策略（masked training），可以在大规模非结构化数据集上实现高质量的新视角生成，且在推理阶段完全去除了对深度估计器的依赖。

### 2. 关键创新与方法论
*   **训练范式变革**：将复杂的“多视角监督”简化为“基于几何先验的自监督”。通过单目深度估计器将单张图片“提升”至3D空间，再进行相机变换投影，生成“伪目标视图”（pseudo-target view）。
*   **掩码训练（Masked Training Formulation）**：这是解决“非配对数据难题”的关键。通过限制几何、感知和纹理损失函数在有效区域内计算，模型能够智能忽略掉投影变换中产生的伪影和遮挡区域（disocclusions）。
*   **推理高效性**：不同于许多生成式模型（如基于NeRF或3D Gaussian Splatting的方法）在推理时需要依赖几何结构，OVIE 在训练后是一个“纯净”的生成网络，推理时无需深度估计器，实现了高达 600 倍的速度提升。

### 3. 对领域的潜在影响
*   **数据效率的跃升**：由于不再依赖多视角图像对（这通常需要复杂的相机阵列拍摄或精准的多视点数据集），OVIE 可以直接利用互联网上海量的“野生”（In-the-wild）图像，这极大地扩展了模型的泛化能力和数据规模（文中提到 3000 万张图像）。
*   **重塑推理架构**：该方法挑战了目前主流的“显式几何（NeRF/3DGS）”范式，证明了在生成式建模中，通过大规模学习隐式表达可以完全替代显式的几何建模，在追求实时渲染的任务中具有极高的竞争力。

### 4. 相关领域与应用前景
*   **内容创作与自动生成**：可用于图像编辑、自动填补背景缺失、提升图像的动态感或立体感，对社交媒体图像生成有直接应用价值。
*   **增强现实（AR）与虚拟现实（VR）**：快速的推理速度使得该技术有望在移动端实时从单图生成场景，实现低成本的 3D 沉浸式体验。
*   **自动驾驶仿真**：利用大量街景单图生成连续视角，为自动驾驶系统提供更丰富的数据增强。

### 5. 可推断的潜在局限性
*   **极端几何畸变的挑战**：虽然采用了掩码训练处理遮挡，但对于大幅度的视角改变（如绕到物体背面），仅靠深度估计器提升的几何 scaffold 必然存在信息缺失，模型在这些区域的补全可能表现为“幻觉”而非真实几何（即：它可能看起来像真的，但几何上未必严谨）。
*   **单目深度估计器的瓶颈**：训练质量高度依赖于预训练深度估计器的准确性。如果深度估计器在特定场景（如透明物体、反光表面）失效，OVIE 的训练可能会引入严重的伪影。
*   **一致性问题**：由于缺乏显式的多视图几何约束，模型在生成连续多帧视频流时，可能会出现时间维度上的抖动或不一致性（Temporal Inconsistency）。

**专家点评**：这篇论文极具启发性。它通过将复杂的3D几何问题转化为大规模的生成学习问题，不仅大幅降低了训练成本，还实现了极佳的零样本（zero-shot）推理性能。这代表了计算机视觉从“精确几何计算”向“大规模统计生成”演进的一种重要趋势。

**Key Findings:**

- Monocular novel-view synthesis has long required multi-view image pairs for supervision, limiting training data scale and diversity.
- We present OVIE, trained entirely on unpaired internet images.
- To handle disocclusions, we introduce a masked training formulation that restricts geometric, perceptual, and textural losses to valid regions, enabling training on 30 million uncurated images.
- Trained exclusively on in-the-wild images, OVIE outperforms prior methods in a zero-shot setting, while being 600x faster than the second-best baseline.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.23488v1)
- [arXiv](https://arxiv.org/abs/2603.23488v1)

---

<a id='2603.23487v1'></a>
## [TETO: Tracking Events with Teacher Observation for Motion Estimation and Frame Interpolation](https://arxiv.org/abs/2603.23487v1)

**Authors:** Jini Yang, Eunbeen Hong, Soowon Son, Hyunkoo Lee, Sunghwan Hong, Sunok Kim, Seungryong Kim

**Published:** 2026-03-24

**Categories:** cs.CV

**Abstract:**

Event cameras capture per-pixel brightness changes with microsecond resolution, offering continuous motion information lost between RGB frames. However, existing event-based motion estimators depend on large-scale synthetic data that often suffers from a significant sim-to-real gap. We propose TETO (Tracking Events with Teacher Observation), a teacher-student framework that learns event motion estimation from only $\sim$25 minutes of unannotated real-world recordings through knowledge distillation from a pretrained RGB tracker. Our motion-aware data curation and query sampling strategy maximizes learning from limited data by disentangling object motion from dominant ego-motion. The resulting estimator jointly predicts point trajectories and dense optical flow, which we leverage as explicit motion priors to condition a pretrained video diffusion transformer for frame interpolation. We achieve state-of-the-art point tracking on EVIMO2 and optical flow on DSEC using orders of magnitude less training data, and demonstrate that accurate motion estimation translates directly to superior frame interpolation quality on BS-ERGB and HQ-EVFI.

**Analysis:**

作为计算机视觉领域的专家，我对这篇论文《TETO: Tracking Events with Teacher Observation》的分析如下：

### 1. 论文核心贡献总结
TETO 提出了一种基于知识蒸馏的教师-学生框架，仅需约 25 分钟的真实世界无标注数据，即可高效训练事件相机（Event Camera）的运动估计模型。该模型通过预测点轨迹和稠密光流，为视频扩散模型提供了高质量的显式运动先验，从而在运动估计和帧插值任务上达到了 SOTA 水准，有效摆脱了对大规模合成数据的依赖。

### 2. 关键创新与方法论
*   **知识蒸馏（Teacher-Student Framework）：** 利用预训练的 RGB 跟踪器作为“教师”，通过其对真实场景的理解来指导“学生”（事件相机模型）学习运动，从而消除了合成数据与真实场景之间的“域差距”（sim-to-real gap）。
*   **运动解耦策略：** 论文提出了一种运动感知的数据筛选（data curation）和查询采样策略，能够将复杂的场景运动显式拆解为目标运动与主导的自运动（ego-motion），提高了模型在有限数据下的学习效率。
*   **多任务联合预测：** 将点轨迹跟踪与稠密光流估计结合，作为一种多尺度运动表达，为下游任务（如视频插值）提供了强大的运动约束，而非仅仅依赖隐式特征表示。

### 3. 对领域的潜在影响
*   **摆脱数据依赖：** 传统事件相机模型高度依赖昂贵的合成数据集（如仿真器生成的数据），TETO 证明了通过小规模真实数据即可实现高性能，极大地降低了数据获取成本。
*   **提升生成式任务的质量：** 过去基于 Transformer 的视频生成模型往往难以处理高速运动，TETO 提供的显式运动先验（Explicit Motion Priors）为视频扩散模型（Video Diffusion Transformer）注入了时空一致性，这在计算机视觉生成任务中是一个重要的方向。

### 4. 受益的相关领域与应用
*   **自动驾驶：** 能够更好地处理高速行驶场景下的运动模糊和光线剧变，提高目标跟踪的鲁棒性。
*   **无人机（UAV）视觉：** 无人机在快速飞行中容易产生剧烈晃动，TETO 的运动解耦能力能够极大地优化自主导航中的视觉里程计。
*   **高帧率视频插值（Slow Motion）：** 针对手机拍摄的事件流，可以合成出比传统方法更平滑、无伪影的高帧率视频。
*   **人机交互（AR/VR）：** 利用事件相机的高时间分辨率特性，实现超低延迟的头戴设备跟踪。

### 5. 可推断的局限性
*   **对教师模型的依赖：** 学生模型的上限很大程度上受限于预训练 RGB 教师模型在极端光照条件（如强逆光或全黑）下的表现，如果教师模型“看不清”，学生模型的学习将面临瓶颈。
*   **计算开销：** 尽管训练数据量少，但推理时需要运行两个模型（事件估计模型 + 视频扩散模型），在边缘设备（Edge Devices）上的实时部署可能仍面临算力挑战。
*   **复杂动态场景的泛化性：** 虽然在 DSEC 和 EVIMO2 上表现出色，但在处理包含大量非刚体运动或极其复杂的遮挡场景时，单一的运动解耦策略是否依然有效，仍需进一步验证。

**专家点评：**
这篇论文的亮点在于其**“反直觉”的效率**——仅通过 25 分钟的真实数据就战胜了以往通过大规模合成数据训练的模型。它巧妙地利用了知识蒸馏来弥合事件数据与 RGB 数据之间的模态鸿沟，这一范式可能会改变未来事件相机在自动驾驶和视频生成领域的研究路径。

**Key Findings:**

- We propose TETO (Tracking Events with Teacher Observation), a teacher-student framework that learns event motion estimation from only $\sim$25 minutes of unannotated real-world recordings through knowledge distillation from a pretrained RGB tracker.
- We achieve state-of-the-art point tracking on EVIMO2 and optical flow on DSEC using orders of magnitude less training data, and demonstrate that accurate motion estimation translates directly to superior frame interpolation quality on BS-ERGB and HQ-EVFI.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.23487v1)
- [arXiv](https://arxiv.org/abs/2603.23487v1)

---

