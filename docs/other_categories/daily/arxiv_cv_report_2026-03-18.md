time: 20260318

# Arxiv Computer Vision Papers - 2026-03-18

## Executive Summary

## Arxiv 计算机视觉领域论文日报执行摘要（2026-03-17）

**1. 核心主题与趋势**

本日论文集体现了计算机视觉领域三个明确的交叉融合趋势：

*   **仿真与数据生成驱动泛化能力**：多篇论文（如 MolmoB0T, ManiTwin, MessyKitchens）聚焦于利用大规模仿真、合成数据或自动化流程来构建数据集或训练系统，旨在解决现实世界中数据稀缺、标注困难的问题，并追求**零样本（Zero-Shot）** 或**少样本**的泛化能力。
*   **3D视觉与具身智能的深度融合**：研究重点明显从传统的2D感知转向与**3D世界交互**。这包括从单目图像进行3D重建与SLAM（M^3），构建交互式3D游戏世界（WorldCam），到面向机器人操作的场景理解（MessyKitchens）、灵巧抓取（DexGrasp-Zero）和视觉语言规划（DreamPlan）。**几何表示（如相机位姿、高斯泼溅）与语义理解的结合**成为关键。
*   **基础模型能力的迁移与专精化**：研究致力于将大规模基础模型（如3D生成模型、多视图基础模型、视频世界模型）的能力，通过重新设计或微调，高效地适配到特定下游任务，例如部件分割（SegviGen）、单目SLAM（M^3）和规划（DreamPlan），体现了“**赋能**”而非“从头训练”的范式。

**2. 突出创新论文**

*   **《MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation》**：其“**大规模仿真实现零样本操控**”的宣称若被验证，将标志着机器人操作范式的重大进步，可能减少对昂贵真实机器人数据的需求。
*   **《M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM》**：创新性地将**密集匹配**、**多视图基础模型**与前沿的**高斯泼溅**3D表示结合，用于单目SLAM，有望在精度、效率和场景理解上取得突破，是技术融合的典范。
*   **《DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping》**：直接挑战机器人学核心难题——**跨不同灵巧手形态的零样本泛化抓取**。其提出的“形态对齐策略”若有效，将极大提升机器人策略的通用性和部署效率。

**3. 新兴研究方向**

*   **“仿真到真实”与数据生成即服务**：以ManiTwin（规模化生成数字物体数据集）为代表，**自动化、高质量、任务导向的合成数据生成**正成为一个独立且关键的研究方向。
*   **以交互和物理为核心的三维场景理解**：MessyKitchens等研究强调对场景中**物体级接触关系、物理属性**的重建与理解，这比传统的静态3D重建更复杂，但对机器人交互至关重要。
*   **视频世界模型作为决策基础**：DreamPlan利用**视频世界模型**进行高效的强化学习微调，将动态预测与规划结合，为复杂的视觉语言决策任务提供了更高效的训练路径。
*   **专用化多模态数据集**：WildDepth针对野生动物3D感知的专用数据集，反映了领域向**细分、挑战性现实场景**深入的趋势。

**4. 推荐精读论文**

根据研究方向的普适性和技术影响力，建议优先阅读：

1.  **M^3 (Dense Matching Meets Multi-View Foundation Models...)**：适合所有关注**3D视觉、SLAM及基础模型应用**的研究者。它是当前技术栈融合的标杆。
2.  **MolmoB0T (Large-Scale Simulation Enables Zero-Shot Manipulation)**：强烈推荐给**机器人学习、具身AI**领域的研究者。其结论对数据驱动机器人学的未来方向有指导意义。
3.  **DexGrasp-Zero (Morphology-Aligned Policy for Zero-Shot Cross-Embodiment...)**：**机器人抓取与强化学习**研究者的必读之作，直接应对部署中的核心泛化挑战。
4.  **DreamPlan (Efficient RL Fine-Tuning of Vision-Language Planners...)**：适合关注**视觉语言动作规划、决策AI**以及**世界模型应用**的研究者，提供了高效的训练框架思路。

**总结**：本日论文整体呈现出计算机视觉研究向**可交互、可泛化、数据高效**的3D具身智能系统快速演进的图景。研究前沿在于如何巧妙地融合基础模型、物理仿真与几何先验，以构建能在复杂现实世界中理解和行动的智能体。

---

## Table of Contents

1. [MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation](#2603.16861v1)
2. [WorldCam: Interactive Autoregressive 3D Gaming Worlds with Camera Pose as a Unifying Geometric Representation](#2603.16871v1)
3. [Demystifing Video Reasoning](#2603.16870v1)
4. [SegviGen: Repurposing 3D Generative Model for Part Segmentation](#2603.16869v1)
5. [MessyKitchens: Contact-rich object-level 3D scene reconstruction](#2603.16868v1)
6. [ManiTwin: Scaling Data-Generation-Ready Digital Object Dataset to 100K](#2603.16866v1)
7. [DreamPlan: Efficient Reinforcement Fine-Tuning of Vision-Language Planners via Video World Models](#2603.16860v1)
8. [M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM](#2603.16844v1)
9. [WildDepth: A Multimodal Dataset for 3D Wildlife Perception and Depth Estimation](#2603.16816v1)
10. [DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping](#2603.16806v1)

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

作为计算机视觉与机器学习领域的专家，我对 **MolmoB0T** 的这项研究进行了深入分析。这篇论文挑战了机器人学习中的一个核心教条：即“模拟器无法完全闭环，必须引入真实世界数据”。

以下是详细分析：

### 1. 核心贡献摘要
该论文证明了通过大规模、多样化的合成训练数据（180万条专家轨迹），机器人策略可以实现无需任何真实数据微调的**零样本（Zero-Shot）跨域迁移**。研究发布了 **MolmoBot-Engine**（Procedural 数据生成流水线）和 **MolmoBot-Data**，并在固定式机械臂和移动操作机器人上验证了其泛化性能，大幅超越了现有的先进模型（如 $\pi_0$）。

### 2. 核心创新与方法论
*   **模拟数据规模与多样性的质变**：该研究不再依赖单一场景，而是利用 **MolmoSpaces** 进行过程化环境生成，涵盖了极高多样性的机器人本体、操作任务及物理场景，证明了“数据规模/多样性”足以抵消“模拟与现实鸿沟（Sim-to-Real Gap）”。
*   **基于 VLM 的策略架构**：将视觉语言模型（Molmo2）作为决策大脑，结合流匹配（Flow-matching）动作头，直接输出控制指令。这种架构能利用 VLM 强大的视觉表征能力来理解零样本场景下的物体关联与空间语义。
*   **三管齐下的架构对比**：对比了基于 VLM 的重型策略（MolmoBot）、复现 $\pi_0$ 的架构（MolmoBot-Pi0）以及适合边缘计算的轻量化策略（MolmoBot-SPOC），为不同落地需求提供了参考。

### 3. 对领域的潜在影响
*   **改变机器人学习的数据范式**：如果“大规模模拟数据”被证明是万能的，那么机器人研究将从“昂贵的真实机器人数据采集”转向“高质量模拟器与数据合成引擎的构建”，极大降低了研究门槛。
*   **VLM 作为机器人的通用“大脑”**：该研究进一步夯实了视觉语言模型作为机器人感知与规划核心组件的地位，展示了通用视觉表征在操纵任务中的直接迁移价值。
*   **基准测试的重构**：其在零样本任务下对 $\pi_0$ 的碾压式表现（79.2% vs 39.2%），可能迫使机器人学习领域重新定义什么是“先进算法”。

### 4. 受益的相关领域与应用
*   **具身智能（Embodied AI）**：对于需要处理高度动态环境的家用机器人和仓储自动化机器人具有直接参考价值。
*   **自动化资产生成**：在自动驾驶、虚拟现实（VR）和游戏AI开发中，其过程化环境生成技术（MolmoBot-Engine）可以快速生成大量的训练数据，减少对人工建模的需求。
*   **边缘 AI**：MolmoBot-SPOC 的设计使其在工业机器人和嵌入式平台上具有广泛的应用前景。

### 5. 可推断的局限性
*   **模拟器的保真度瓶颈**：尽管论文宣称零样本迁移有效，但在极度复杂、具有挑战性的物理接触（如复杂的摩擦力、软体变形）或非结构化环境（如杂乱无章的真实实验室）中，模拟器的物理建模仍可能出现偏差。
*   **任务类型的扩展性**：目前主要针对“抓取与操作（Pick-and-Place/Articulated）”，对于需要精细力反馈、长周期推理或极端环境适应性的任务，仅靠模拟训练是否足够仍待验证。
*   **计算资源需求**：尽管政策本身可能轻量，但支撑 180 万条专家轨迹的生成和训练过程需要极其庞大的算力，这对中小型研究机构构成门槛。

### 专家点评（为什么这很有趣）：
在 CV 领域，我们见证了从 ImageNet 到大规模 Web 数据的飞跃带来了模型能力的质变。**MolmoB0T 的意义在于，它试图在机器人领域复现这一逻辑**——即通过“模拟器的规模化”来解决“具身智能的感知/决策泛化问题”。如果它成功了，这意味着我们可能不再需要像 Data-hungry 的大模型那样收集数万小时的真实视频，而是可以通过构建一个“无限的模拟器”来让机器人在现实中无缝作业。这是一场关于**数据本质**的范式转移。

**Key Findings:**

- With sufficiently large-scale and diverse simulated synthetic training data, we show that zero-shot transfer to the real world is not only possible, but effective for both static and mobile manipulation.
- We introduce MolmoBot-Engine, a fully open-source pipeline for procedural data generation across robots, tasks, and diverse simulated environments in MolmoSpaces.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16861v1)
- [arXiv](https://arxiv.org/abs/2603.16861v1)

---

<a id='2603.16871v1'></a>
## [WorldCam: Interactive Autoregressive 3D Gaming Worlds with Camera Pose as a Unifying Geometric Representation](https://arxiv.org/abs/2603.16871v1)

**Authors:** Jisu Nam, Yicong Hong, Chun-Hao Paul Huang, Feng Liu, JoungBin Lee, Jiyoung Kim, Siyoon Jin, Yunsung Lee, Jaeyoon Jung, Suhwan Choi, Seungryong Kim, Yang Zhou

**Published:** 2026-03-17

**Categories:** cs.CV

**Abstract:**

Recent advances in video diffusion transformers have enabled interactive gaming world models that allow users to explore generated environments over extended horizons. However, existing approaches struggle with precise action control and long-horizon 3D consistency. Most prior works treat user actions as abstract conditioning signals, overlooking the fundamental geometric coupling between actions and the 3D world, whereby actions induce relative camera motions that accumulate into a global camera pose within a 3D world. In this paper, we establish camera pose as a unifying geometric representation to jointly ground immediate action control and long-term 3D consistency. First, we define a physics-based continuous action space and represent user inputs in the Lie algebra to derive precise 6-DoF camera poses, which are injected into the generative model via a camera embedder to ensure accurate action alignment. Second, we use global camera poses as spatial indices to retrieve relevant past observations, enabling geometrically consistent revisiting of locations during long-horizon navigation. To support this research, we introduce a large-scale dataset comprising 3,000 minutes of authentic human gameplay annotated with camera trajectories and textual descriptions. Extensive experiments show that our approach substantially outperforms state-of-the-art interactive gaming world models in action controllability, long-horizon visual quality, and 3D spatial consistency.

**Analysis:**

### 1. 摘要翻译
**WorldCam**：一种通过相机姿态作为统一几何表示的交互式自回归3D游戏世界模型。
近期的视频扩散Transformer实现了交互式游戏世界模型，允许用户在扩展的时间跨度内探索生成环境。然而，现有方法在精确的动作控制和长期3D一致性方面存在困难。大多数先验研究将用户动作视为抽象的条件信号，忽略了动作与3D世界之间的基础几何耦合关系。本文确立了“相机姿态”作为统一的几何表示，以联合基础（Ground）即时动作控制与长期3D一致性。首先，我们定义了一个基于物理的连续动作空间，并将用户输入表示为李代数（Lie algebra），从而推导出精确的6自由度相机姿态，将其通过相机嵌入器注入生成模型，确保动作对齐。其次，我们利用全局相机姿态作为空间索引，检索相关的过去观察结果，实现长程导航中对位置的几何一致性重访。为此，我们引入了一个包含3,000分钟真实人类游戏记录的大规模数据集，并标注了相机轨迹和文本描述。实验证明，我们的方法在动作可控性、长期视觉质量和3D空间一致性方面大幅超越了现有的先进模型。

### 2. 方法动机分析
- **驱动力**：现有的交互式游戏模型倾向于将键盘/鼠标输入视为抽象控制信号，导致生成结果在长程交互中丧失物理约束，出现“漂移”和“几何坍塌”。
- **痛点**：先前工作（如GameCraft）使用线性近似处理动作，无法捕捉相机运动的耦合动力学（如“螺钉运动”），且缺乏长程记忆机制来保证多次往返同一空间时的视觉一致性。
- **研究假设**：动作与3D世界的耦合完全由“相机姿态”这一中间桥梁决定。只要在李群/李代数空间精确建模相机运动，并以全局相机姿态为索引检索记忆，即可实现长期一致的交互。

### 3. 方法设计详解
- **核心流程**：
  1. **动作-相机映射**：将键盘/鼠标输入转化为6自由度的 twist 向量，通过指数映射（Exponential Map）在 $SE(3)$ 流形上计算相对相机姿态 $\Delta P_i$，而非线性独立更新位移和旋转。
  2. **相机可控生成**：通过 MLP 将相对姿态编码为 Plücker 嵌入，注入视频 DiT 的中间层。
  3. **记忆检索与锚定**：维护一个包含过去 latents 及其对应全局姿态的记忆池。利用当前时刻的全局相机姿态作为查询，检索历史记忆，并将其拼接到当前去噪窗口中，以此强制空间几何一致性。
  4. **渐进式推理**：通过在去噪窗口内引入不同噪声阶段，实现长程稳定生成。

### 4. 方法对比分析
- **本质区别**：从传统的“动作->视频”端到端映射，转变为“动作->相机姿态->几何一致的视频生成”。
- **创新点**：
  - 引入基于李代数的动作参数化，精确建模复杂曲线运动（Screw motion）。
  - 使用全局相机姿态作为空间索引进行记忆检索，而非单纯的基于内容的检索。
- **适用场景**：高交互性、需要长程导航及复杂摄像机位移的游戏环境。

### 5. 实验分析
- **验证方法**：在WorldCam-50h数据集上进行测试，对比Yume、Matrix-Game 2.0等模型。
- **关键结果**：在动作控制精度（RPE）、长程视觉质量（VBench指标）和几何一致性（MEt3R）上均显著领先。
- **优势**：极高的几何准确度，长程漂移极小，支持闭环导航。
- **局限**：推理效率受限于多步采样，目前尚需优化以达到实时推理。

### 6. 实用指南
- **开源情况**：论文明确表示将公开代码与数据集。
- **实现细节**：建议重点参考公式(2)的矩阵指数映射实现，以及公式(6)(7)的记忆库分层检索逻辑。
- **迁移可能**：该框架的“相机姿态统一表示”思想可直接迁移至机器人仿真、具身智能及各种基于视频的虚拟环境构建任务。

### 7. 总结
- **核心思想**：以相机姿态为几何锚点，构建具备物理一致性的交互世界模型。
- **速记版pipeline**：
  1. 动作转李代数姿态；
  2. 姿态嵌入注入DiT模型；
  3. 全局坐标记忆检索；
  4. 渐进式多阶段去噪。

**Key Findings:**

- To support this research, we introduce a large-scale dataset comprising 3,000 minutes of authentic human gameplay annotated with camera trajectories and textual descriptions.
- Extensive experiments show that our approach substantially outperforms state-of-the-art interactive gaming world models in action controllability, long-horizon visual quality, and 3D spatial consistency.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16871v1)
- [arXiv](https://arxiv.org/abs/2603.16871v1)

---

<a id='2603.16870v1'></a>
## [Demystifing Video Reasoning](https://arxiv.org/abs/2603.16870v1)

**Authors:** Ruisi Wang, Zhongang Cai, Fanyi Pu, Junxiang Xu, Wanqi Yin, Maijunxian Wang, Ran Ji, Chenyang Gu, Bo Li, Ziqi Huang, Hokin Deng, Dahua Lin, Ziwei Liu, Lei Yang

**Published:** 2026-03-17

**Categories:** cs.CV, cs.AI

**Abstract:**

Recent advances in video generation have revealed an unexpected phenomenon: diffusion-based video models exhibit non-trivial reasoning capabilities. Prior work attributes this to a Chain-of-Frames (CoF) mechanism, where reasoning is assumed to unfold sequentially across video frames. In this work, we challenge this assumption and uncover a fundamentally different mechanism. We show that reasoning in video models instead primarily emerges along the diffusion denoising steps. Through qualitative analysis and targeted probing experiments, we find that models explore multiple candidate solutions in early denoising steps and progressively converge to a final answer, a process we term Chain-of-Steps (CoS). Beyond this core mechanism, we identify several emergent reasoning behaviors critical to model performance: (1) working memory, enabling persistent reference; (2) self-correction and enhancement, allowing recovery from incorrect intermediate solutions; and (3) perception before action, where early steps establish semantic grounding and later steps perform structured manipulation. During a diffusion step, we further uncover self-evolved functional specialization within Diffusion Transformers, where early layers encode dense perceptual structure, middle layers execute reasoning, and later layers consolidate latent representations. Motivated by these insights, we present a simple training-free strategy as a proof-of-concept, demonstrating how reasoning can be improved by ensembling latent trajectories from identical models with different random seeds. Overall, our work provides a systematic understanding of how reasoning emerges in video generation models, offering a foundation to guide future research in better exploiting the inherent reasoning dynamics of video models as a new substrate for intelligence.

**Analysis:**

这是一份关于《Demystifying Video Reasoning》（解构视频推理）的深度技术分析报告。

### 1. 摘要翻译
近期视频生成模型展现出惊人的非凡推理能力。既往研究将其归因为“帧链（CoF）”机制，认为推理是跨视频帧依次展开的。本研究对该假设提出质疑，揭示了一种根本不同的机制：视频模型中的推理主要沿扩散去噪步骤产生。通过定性分析和针对性实验，我们发现模型在去噪早期探索多个候选解，并随去噪推进逐渐收敛至最终答案，我们称之为“步链（CoS）”。除该核心机制外，我们还识别出几种对模型性能至关重要的推理行为：(1) 工作记忆，支持持续参照；(2) 自我修正与增强，允许从错误中间解中恢复；(3) 先感知后动作，早期步骤负责语义定位，后期执行结构化操作。在扩散步骤内部，我们揭示了扩散Transformer中自进化的功能专门化。受此启发，我们提出了一个简单的训练后推理策略，证明通过对来自相同模型不同随机种子的潜在轨迹进行集成，可以提升推理效果。

### 2. 方法动机分析
*   **驱动力**：旨在填补视频生成模型在“推理机制”上的黑盒空白。打破主流关于“推理在时间维度（帧间）发生”的认知。
*   **痛点**：既往研究（CoF假设）难以解释模型如何在复杂逻辑任务中进行长程一致性规划。
*   **核心直觉（研究假设）**：视频推理的本质不是帧与帧的因果累积，而是去噪过程中从“噪声-概率解空间”到“确定性解空间”的收敛过程。

### 3. 方法设计详解
*   **核心机制 (Chain-of-Steps, CoS)**：
    *   **多路径探索**：在去噪初期，模型隐式地在潜在空间中对多种可能路径进行“宽度优先”式的同步模拟。
    *   **逐步修剪**：随着扩散步数增加，原本概率较低的候选解（错误分支）通过去噪梯度被逐渐抑制。
*   **功能专门化 (Layer-wise Specialization)**：
    *   **早期层 (0-9)**：聚焦于全局结构、背景与基本语义定位（感知阶段）。
    *   **中期层 (10-29)**：执行推理核心逻辑，处理目标物体的运动和交互。
    *   **后期层 (30-39)**：将推理结果具象化，合并生成最终视频状态。
*   **训练无损集成策略 (Training-free Ensemble)**：
    *   执行三次基于不同随机噪声种子的推理前向传播。
    *   在关键的“推理窗”（第20-29层）对隐藏状态进行空间-时间平均。
    *   **意义**：通过投票机制过滤种子引起的随机噪声，保留推理的共性逻辑结构。

### 4. 方法对比分析
*   **本质区别**：从“时间跨度上的序列推理（CoF）”转向“生成过程中的迭代收敛推理（CoS）”。
*   **创新贡献**：首次证明视频生成模型在推理时本质上是“并行搜索”而非“序列执行”。
*   **适用场景**：适用于所有基于扩散（Diffusion）架构的视频生成模型，尤其在需要物理遵循、逻辑推理的任务中表现显著。

### 5. 实验分析
*   **验证方法**：通过“Noise at Step”（步级噪声注入）与“Noise at Frame”（帧级噪声注入）对比实验，证明系统对去噪步骤的敏感度远高于对特定帧的敏感度。
*   **关键结论**：破坏早期去噪步骤（步骤20-30）会导致 reasoning 彻底失效，证明推理过程是分阶段收敛的。
*   **局限性**：在大规模视频生成中，集成多个路径会带来成倍的推理时间开销（虽无需额外训练）。

### 6. 实用指南
*   **开源与复现**：基于VBVR-Wan2.2模型，作者建议在推理阶段调用多个噪声种子。
*   **实现要点**：关键在于寻找模型的“推理活跃窗”，不同结构的DiT模型（如ViT-S vs ViT-L）该窗口可能不同，建议通过激活图可视化（Activation Visualization）确认。
*   **迁移**：该“潜在空间集成法”可直接迁移至任何基于Transformer的扩散模型，无需微调。

### 7. 总结
*   **核心思想**：推理是通过去噪迭代收敛的概率搜索过程。
*   **速记版pipeline**：
    1. 同一提示词运行多次推理；
    2. 提取中期网络层的隐藏特征；
    3. 在潜在空间对多路径特征求平均；
    4. 将聚合后的状态作为后续推理的引导。

**Key Findings:**

- We show that reasoning in video models instead primarily emerges along the diffusion denoising steps.
- Motivated by these insights, we present a simple training-free strategy as a proof-of-concept, demonstrating how reasoning can be improved by ensembling latent trajectories from identical models with different random seeds.
- Overall, our work provides a systematic understanding of how reasoning emerges in video generation models, offering a foundation to guide future research in better exploiting the inherent reasoning dynamics of video models as a new substrate for intelligence.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16870v1)
- [arXiv](https://arxiv.org/abs/2603.16870v1)

---

<a id='2603.16869v1'></a>
## [SegviGen: Repurposing 3D Generative Model for Part Segmentation](https://arxiv.org/abs/2603.16869v1)

**Authors:** Lin Li, Haoran Feng, Zehuan Huang, Haohua Chen, Wenbo Nie, Shaohua Hou, Keqing Fan, Pan Hu, Sheng Wang, Buyu Li, Lu Sheng

**Published:** 2026-03-17

**Categories:** cs.CV

**Abstract:**

We introduce SegviGen, a framework that repurposes native 3D generative models for 3D part segmentation. Existing pipelines either lift strong 2D priors into 3D via distillation or multi-view mask aggregation, often suffering from cross-view inconsistency and blurred boundaries, or explore native 3D discriminative segmentation, which typically requires large-scale annotated 3D data and substantial training resources. In contrast, SegviGen leverages the structured priors encoded in pretrained 3D generative model to induce segmentation through distinctive part colorization, establishing a novel and efficient framework for part segmentation. Specifically, SegviGen encodes a 3D asset and predicts part-indicative colors on active voxels of a geometry-aligned reconstruction. It supports interactive part segmentation, full segmentation, and full segmentation with 2D guidance in a unified framework. Extensive experiments show that SegviGen improves over the prior state of the art by 40% on interactive part segmentation and by 15% on full segmentation, while using only 0.32% of the labeled training data. It demonstrates that pretrained 3D generative priors transfer effectively to 3D part segmentation, enabling strong performance with limited supervision. See our project page at https://fenghora.github.io/SegviGen-Page/.

**Analysis:**

这是一份关于《SegviGen: Repurposing 3D Generative Model for Part Segmentation》的深入技术分析。

---

### 1. 摘要翻译
我们提出了 SegviGen，一个将原生 3D 生成模型重新用于 3D 部件分割的框架。现有流水线要么通过蒸馏或多视图掩码聚合将强大的 2D 先验提升至 3D，往往受到跨视图不一致和边界模糊的困扰；要么探索原生 3D 判别式分割，这通常需要大规模标注的 3D 数据和巨大的训练资源。相比之下，SegviGen 利用预训练 3D 生成模型中编码的结构先验，通过独特的部件着色来诱导分割。具体而言，SegviGen 对 3D 资产进行编码，并在几何对齐的重建体素上预测部件指示颜色。它在统一框架内支持交互式部件分割、全分割以及带 2D 指导的全分割。实验表明，SegviGen 在交互式部件分割上较现有最先进水平提升了 40%，在全分割上提升了 15%，且仅使用了 0.32% 的标注训练数据。

### 2. 方法动机分析
*   **驱动力**：通过复用大规模 3D 生成模型已内化的丰富几何与纹理先验，以实现极高的数据效率（Data Efficiency）和更好的部件分割质量。
*   **现有方法痛点**：
    *   **2D-to-3D 提升**：依赖 2D 模型投影，存在跨视图不一致性和多视图融合带来的边界模糊。
    *   **原生 3D 判别式模型**：过度依赖大规模、高质量、且定义不一致的 3D 部件标注数据，导致泛化能力弱。
*   **研究假设**：3D 生成模型（如 TRELLIS2）在训练过程中通过去噪任务学习到的 latent 空间，本身蕴含了物体精细的结构与部件信息。将“分割任务”重构为“色彩预测”任务，可以直接“唤醒”生成模型中的部件先验。

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入**：3D 网格模型，以及可选的用户交互点或 2D 引导图。
    2.  **编码**：使用预训练的 3D VAE 将输入编码为结构化 latent $z$。
    3.  **多任务条件化**：将任务类型（交互式/全分割）、交互点（Point Embedding）或 2D 引导图（Image Encoder）编码为条件 token。
    4.  **去噪预测**：模型在 latent 空间进行流匹配（Flow Matching），预测噪声残差，同时实现几何重建与部件颜色预测。
    5.  **输出**：部件着色的 3D 体素，颜色聚类后即为分割结果。
*   **模型结构**：基于 DiT (Diffusion Transformer) 架构，额外引入任务嵌入（Task Embedding）以实现多任务的统一管理。
*   **关键公式**：$\hat{v}_\theta = f_\theta(y_t, z, C, e_\tau, t)$，其中 $y_t$ 为带有噪声的部件颜色 latent，$z$ 为几何 latent，$C$ 为点或图引导，$e_\tau$ 为任务索引嵌入。该结构将部件属性作为“颜色”处理，完美契合生成模型的输出层。

### 4. 方法对比分析
*   **本质区别**：不将分割作为额外的分类任务，而是视为生成模型的一种特定“着色”输出，无需专门设计分割 Head。
*   **创新贡献**：提出了一种通用的、任务无关的 3D 分割框架，打破了对大规模 3D 标注的依赖。
*   **适用场景**：极低样本量（Few-shot/Zero-shot）下的精细化部件分割，尤其适用于工业 3D 编辑场景。

### 5. 实验分析
*   **验证方法**：在 PartObjaverse-Tiny 和 PartNeXT 上进行交互式和全分割测试。
*   **关键结果**：仅用 0.32% 的训练数据，交互任务 IoU@1 提升 40%，全分割任务提升 15%。
*   **主要优势**：极佳的边界清晰度，强大的类别泛化能力，以及通过 2D 引导实现的可控性。
*   **主要局限**：推理过程仍依赖 12 步采样（虽然已优化），在极高实时性要求下的端侧部署存在挑战。

### 6. 实用指南
*   **开源情况**：官方提供了项目页面（https://fenghora.github.io/SegviGen-Page/），基于 TRELLIS2 进行开发。
*   **实现细节**：建议训练时采用 $K=10$ 的色板采样策略来减轻模型对特定色彩的过拟合。
*   **迁移可能**：该框架核心是“任务重构”，可直接迁移至其他以体素或点云为输出形式的 3D 属性预测任务（如材质编辑、语义部件生成）。

### 7. 总结
*   **核心思想**：将 3D 分割转化为生成模型的部件色彩填充任务。
*   **速记版 Pipeline**：
    1. 冻结预训练 3D 生成模型权重；
    2. 将部件标签转换为彩色 latent；
    3. 将任务指令与 geometry latent 拼合；
    4. 训练模型预测彩色残差，完成着色分割。

**Key Findings:**

- We introduce SegviGen, a framework that repurposes native 3D generative models for 3D part segmentation.
- In contrast, SegviGen leverages the structured priors encoded in pretrained 3D generative model to induce segmentation through distinctive part colorization, establishing a novel and efficient framework for part segmentation.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16869v1)
- [arXiv](https://arxiv.org/abs/2603.16869v1)

---

<a id='2603.16868v1'></a>
## [MessyKitchens: Contact-rich object-level 3D scene reconstruction](https://arxiv.org/abs/2603.16868v1)

**Authors:** Junaid Ahmed Ansari, Ran Ding, Fabio Pizzati, Ivan Laptev

**Published:** 2026-03-17

**Categories:** cs.CV, cs.AI, cs.RO

**Abstract:**

Monocular 3D scene reconstruction has recently seen significant progress. Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image. Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations. Notably, beyond shape and pose estimation of individual objects, applications in robotics and animation require physically-plausible scene reconstruction where objects obey physical principles of non-penetration and realistic contacts. In this work we advance object-level scene reconstruction along two directions. First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate object contacts. Second, we build on the recent SAM 3D approach for single-object reconstruction and extend it with Multi-Object Decoder (MOD) for joint object-level scene reconstruction. To validate our contributions, we demonstrate MessyKitchens to significantly improve previous datasets in registration accuracy and inter-object penetration. We also compare our multi-object reconstruction approach on three datasets and demonstrate consistent and significant improvements of MOD over the state of the art. Our new benchmark, code and pre-trained models will become publicly available on our project website: https://messykitchens.github.io/.

**Analysis:**

以下是对该论文的深度分析：

### 1. 摘要翻译
单目3D场景重建已取得显著进展，但将复杂场景分解为独立3D对象仍具挑战。为满足机器人与动画领域对物理可解释性（非穿透、真实接触）的需求，本文提出：1. **MessyKitchens**，一个包含真实世界杂乱场景、高保真对象级3D真值及准确接触信息的全新数据集；2. **多对象解码器 (MOD)**，一种在SAM 3D基础上扩展的架构，通过联合建模实现对象级场景重建。实验表明，MOD在注册精度与对象间穿透问题上均优于现有技术。

### 2. 方法动机分析
*   **驱动力**：现有的对象级重建方法大多将对象视为独立个体，忽略了场景中对象间的物理空间约束（如非穿透性），导致重建出的场景在物理上往往是“漂浮”或相互穿透的，无法满足机器人操作需求。
*   **现有痛点**：缺乏真实且杂乱场景下的高精度3D接触真值，且主流方法（如SAM 3D）缺乏端到端的多对象协同推理机制。
*   **研究假设**：通过在重建过程中引入对象间的上下文交互（通过注意力机制），可以修正独立对象预测的偏差，从而实现全局空间一致且物理可行的重建。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **输入**：RGB图像及各对象的2D掩码（Mask）。
    2.  **基座**：使用SAM 3D提取形状（Shape tokens）和位姿（Pose tokens）。
    3.  **MOD推理（核心）**：输入所有对象的Tokens，通过$K$个Transformer模块进行多对象自注意力（处理位姿相关性）和多对象交叉注意力（将位姿Token与形状Token关联，实现几何引导的位姿修正）。
    4.  **输出**：得到一组残差位姿更新量($\tilde{\mathbf{P}}$)，将其叠加到原始预测位姿上，实现最终的场景感知修正。
*   **算法逻辑**：MOD不仅是在学习单个对象的位姿，而是通过$K$层堆叠，将场景中所有对象的几何特征视为上下文（Keys/Values），共同决定每个对象的最终位姿。这有效解决了物理遮挡带来的独立估计失效问题。

### 4. 方法对比分析
*   **本质区别**：从“独立估计”转变为“场景联合优化”，通过注意力机制捕捉了对象间的空间依赖。
*   **创新贡献**：提出了一种无需复杂物理引擎、仅靠注意力机制学习物理约束的位姿修正模块；构建了高精度接触感知的基准测试集。
*   **适用场景**：复杂杂乱环境（如厨房、实验室）的机器人视觉感知与操作任务。

### 5. 实验分析
*   **关键结论**：在MessyKitchens及GraspNet-1B、HouseCat6D数据集上，MOD在场景级重建（IoU提升、Chamfer Distance降低）方面表现均优于SAM 3D等基线。
*   **优势**：在杂乱场景中显著降低了对象间穿透面积，提升了全局一致性。
*   **局限**：对严重遮挡的边缘部分恢复仍依赖于SAM 3D本身的特征提取能力；推理增加了一定计算开销。

### 6. 实用指南
*   **开源情况**：项目主页为 https://messykitchens.github.io/。
*   **实现要点**：关键超参数 $K=3$（Transformer层数），训练需注意使用该文定义的包含旋转、平移和比例损失（以及对偶覆盖下的四元数校准）的复合损失函数。
*   **迁移建议**：MOD架构非常通用，可作为“后处理器”插件插入到任何基于Token的单对象重建模型中，只需提供对象的初始位姿和形状编码。

### 7. 总结
*   **核心思想**：利用Transformer的多对象注意力机制实现物理一致的3D位姿修正。
*   **速记版Pipeline**：
    1. 使用SAM 3D提取各对象位姿和形状。
    2. 将位姿特征进行跨对象注意力交互。
    3. 结合形状信息对位姿进行物理修正。
    4. 输出最终准确且非穿透的全局位姿。

**Key Findings:**

- First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate object contacts.
- To validate our contributions, we demonstrate MessyKitchens to significantly improve previous datasets in registration accuracy and inter-object penetration.
- Our new benchmark, code and pre-trained models will become publicly available on our project website: https://messykitchens.github.io/.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16868v1)
- [arXiv](https://arxiv.org/abs/2603.16868v1)

---

<a id='2603.16866v1'></a>
## [ManiTwin: Scaling Data-Generation-Ready Digital Object Dataset to 100K](https://arxiv.org/abs/2603.16866v1)

**Authors:** Kaixuan Wang, Tianxing Chen, Jiawei Liu, Honghao Su, Shaolong Zhu, Minxuan Wang, Zixuan Li, Yue Chen, Huan-ang Gao, Yusen Qin, Jiawei Wang, Qixuan Zhang, Lan Xu, Jingyi Yu, Yao Mu, Ping Luo

**Published:** 2026-03-17

**Categories:** cs.RO, cs.AI, cs.GR, cs.LG, cs.SE

**Abstract:**

Learning in simulation provides a useful foundation for scaling robotic manipulation capabilities. However, this paradigm often suffers from a lack of data-generation-ready digital assets, in both scale and diversity. In this work, we present ManiTwin, an automated and efficient pipeline for generating data-generation-ready digital object twins. Our pipeline transforms a single image into simulation-ready and semantically annotated 3D asset, enabling large-scale robotic manipulation data generation. Using this pipeline, we construct ManiTwin-100K, a dataset containing 100K high-quality annotated 3D assets. Each asset is equipped with physical properties, language descriptions, functional annotations, and verified manipulation proposals. Experiments demonstrate that ManiTwin provides an efficient asset synthesis and annotation workflow, and that ManiTwin-100K offers high-quality and diverse assets for manipulation data generation, random scene synthesis, and VQA data generation, establishing a strong foundation for scalable simulation data synthesis and policy learning. Our webpage is available at https://manitwin.github.io/.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对 **ManiTwin** 这篇论文的分析如下：

### 1. 核心贡献总结
ManiTwin 提出了一套自动化的高效管线，实现了从单张图像到“数据生成就绪型”（data-generation-ready）3D数字孪生体的转化。该工作构建了包含 100K 个具备物理属性、语义标注、语言描述及操作建议的高质量 3D 数据集（ManiTwin-100K），极大地缓解了机器人模拟学习中高质量训练数据匮乏的痛点。

### 2. 核心创新与方法论
*   **端到端的自动化流程**：该管线解决了从 2D 到 3D 重建中不仅要“形似”还要“用得上”的难题。不仅生成几何模型，更关键的是赋予了模型**物理属性、功能语义及操作可行性分析**。
*   **规模化与多样性**：通过自动化流程将数据集规模提升至 10 万量级，打破了以往手工构建 3D 资产库的效率瓶颈。
*   **多维度标注集成**：模型不仅仅是视觉上的 3D 网格，还包含了机器人在模拟环境（Simulation）中操作所需的核心元数据，这使得数据能直接对接强化学习（RL）和模仿学习（IL）的训练环境。

### 3. 对领域的潜在影响
*   **弥合模拟与现实（Sim-to-Real）的鸿沟**：通过海量且多样化的 3D 数据生成，机器人策略学习可以覆盖更广泛的场景分布，从而提高政策的鲁棒性和泛化能力。
*   **标准化数据生产范式**：ManiTwin 为机器人领域建立了一个“数据集即服务”（Dataset-as-a-Service）的模板，可能会改变未来机器人社区构建仿真环境的方式。
*   **促进具身智能（Embodied AI）发展**：为训练大规模具身智能模型提供了底层数据基石，使机器人能够像 LLM 学习文本一样，在海量虚拟交互中学习物理世界规律。

### 4. 相关领域与受益应用
*   **机器人操作（Robotic Manipulation）**：直接为抓取、堆叠、工具使用等任务提供仿真训练数据。
*   **视觉问答（VQA）与多模态感知**：利用生成的 3D 场景与语言描述对，训练多模态大模型理解物理世界的空间逻辑。
*   **数字孪生与元宇宙**：高效构建可交互的高质量 3D 资产，适用于游戏开发、虚拟现实及工业仿真。
*   **随机场景合成（Random Scene Synthesis）**：为计算机视觉算法提供大规模的分布外（OOD）测试用例。

### 5. 潜在局限性（基于摘要的推理）
*   **“Sim-to-Real”的迁移误差**：虽然提供了物理属性，但生成的物理参数（摩擦系数、质量分布等）是否与现实物体完全一致，仍需在真实硬件上验证。
*   **语义与几何质量的边界**：在 10 万量级的自动化生产过程中，几何细节（如精细的结构）和语义标注的准确性是否会存在长尾误差（例如复杂铰链结构或特殊材质物体的建模效果）。
*   **计算成本与实时性**：虽然管线高效，但对于终端用户而言，将这 10 万个资产集成到特定的机器人仿真框架中（如 Isaac Gym, MuJoCo）是否需要大量额外的适配工作。

**专家视角点评**：
这篇论文的趣味性在于它巧妙地抓住了“具身智能数据荒”这一关键痛点。过去我们往往依赖于手工精心设计的少量资产，或者质量参差不齐的合成数据，而 **ManiTwin 提供了一个“量变引起质变”的路径**。它不仅是 3D 重建的改进，更是将“视觉感知”与“物理行为”在数据层面完成了深度融合，是实现具身智能规模化扩容的重要基石。

**Key Findings:**

- In this work, we present ManiTwin, an automated and efficient pipeline for generating data-generation-ready digital object twins.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16866v1)
- [arXiv](https://arxiv.org/abs/2603.16866v1)

---

<a id='2603.16860v1'></a>
## [DreamPlan: Efficient Reinforcement Fine-Tuning of Vision-Language Planners via Video World Models](https://arxiv.org/abs/2603.16860v1)

**Authors:** Emily Yue-Ting Jia, Weiduo Yuan, Tianheng Shi, Vitor Guizilini, Jiageng Mao, Yue Wang

**Published:** 2026-03-17

**Categories:** cs.RO

**Abstract:**

Robotic manipulation requires sophisticated commonsense reasoning, a capability naturally possessed by large-scale Vision-Language Models (VLMs). While VLMs show promise as zero-shot planners, their lack of grounded physical understanding often leads to compounding errors and low success rates when deployed in complex real-world environments, particularly for challenging tasks like deformable object manipulation. Although Reinforcement Learning (RL) can adapt these planners to specific task dynamics, directly fine-tuning VLMs via real-world interaction is prohibitively expensive, unsafe, and sample-inefficient. To overcome this bottleneck, we introduce DreamPlan, a novel framework for the reinforcement fine-tuning of VLM planners via video world models. Instead of relying on costly physical rollouts, DreamPlan first leverages the zero-shot VLM to collect exploratory interaction data. We demonstrate that this sub-optimal data is sufficient to train an action-conditioned video generation model, which implicitly captures complex real-world physics. Subsequently, the VLM planner is fine-tuned entirely within the "imagination" of this video world model using Odds Ratio Policy Optimization (ORPO). By utilizing these virtual rollouts, physical and task-specific knowledge is efficiently injected into the VLM. Our results indicate that DreamPlan bridges the gap between semantic reasoning and physical grounding, significantly improving manipulation success rates without the need for large-scale real-world data collection. Our project page is https://psi-lab.ai/DreamPlan/.

**Analysis:**

这是一篇关于通过视频世界模型实现高效强化学习微调视觉语言模型（VLM）以解决复杂可变形物体操纵任务的深度论文分析。

### 1. 摘要翻译
机器人操纵需要复杂的常识推理，视觉语言模型（VLM）作为零样本规划器极具潜力。然而，VLM缺乏针对性的物理感知，在复杂环境（尤其是可变形物体操纵）中往往表现出物理上无效的决策。尽管强化学习（RL）可以调整模型，但在真实环境中的物理交互既昂贵又不安全。本文提出了**DreamPlan**，这是一个通过视频世界模型进行强化微调的新框架。DreamPlan利用零样本VLM收集探索性交互数据，并据此训练一个动作条件下的视频世界模型，该模型隐式捕捉了复杂的物理动力学。随后，VLM规划器在世界模型的“想象”空间中，通过胜率策略优化（ORPO）进行离线微调。结果表明，DreamPlan成功弥合了语义推理与物理接地之间的差距，显著提升了操纵成功率，且无需大规模真实世界数据采集。

### 2. 方法动机分析
*   **驱动力**：旨在解决零样本VLM规划器在处理可变形物体（如布料、绳索）时，“语义合理但物理无效”的现实鸿沟。
*   **现有方法痛点**：传统RL依赖大规模真实物理交互，不仅昂贵且效率极低；传统仿真器在处理复杂可变形物体时存在巨大的仿真到真实（Sim-to-Real）差距。
*   **研究假设**：通过VLM收集的少量非最优探索数据，足以训练一个能够预测动作结果的视频世界模型，该模型可作为离线微调的“虚拟实验室”。

### 3. 方法设计详解
**DreamPlan Pipeline**：
1.  **零样本探索**：利用预训练VLM通过随机/启发式策略在真实环境进行初步交互，收集轨迹数据。
2.  **世界模型训练**：使用视频扩散模型（CogVideoX-5B）作为主干，引入ControlNet将动作序列渲染为视频结构条件，通过预测动作后的物体变形，学习环境动力学。
3.  **想象空间优化**：在模型“想象”中，通过**Best-of-K**采样生成多个动作假设，对比结果与目标图像，挑选最优行为作为正样本，其余为负样本，利用**ORPO（Odds Ratio Policy Optimization）**直接微调VLM策略。

**关键组件**：
*   **ControlNet架构**：将渲染后的机器人末端运动轨迹作为结构化条件，强制视频模型将注意力集中在“动作-物体”交互上，而非背景细节。
*   **Best-of-K + ORPO**：这是核心创新点。它将昂贵的扩散模型推理与策略优化解耦，避免了在RL梯度循环中频繁调用生成模型。

### 4. 方法对比分析
*   **本质区别**：不直接在RL循环中生成视频（计算太慢），而是通过世界模型生成离线对比偏好数据，再进行偏好驱动的离线微调。
*   **创新贡献**：提出了一种基于物理感知生成的“想象微调”闭环，显著降低了策略微调的计算开销。
*   **适用场景**：高动态、难建模的可变形物体 manipulation 任务（如布料整理、绳索解结等）。

### 5. 实验分析（精简版）
*   **验证方法**：在绳索、布料、玩具摆弄三个任务上进行真实物理实验。
*   **关键结果**：在成功率指标上，DreamPlan相较于零样本基线提升了15%-40%；在推理效率上，单次决策仅需约1秒。
*   **优势**：极高的数据效率和计算效率；赋予了模型所需的物理“预见性”。
*   **局限**：模型对长时序复杂逻辑的物理预测准确度仍受扩散模型长时一致性限制。

### 6. 实用指南
*   **开源**：项目主页：https://psi-lab.ai/DreamPlan/
*   **实现细节**：建议在视频生成端采用“只裁剪物体（Object-only）”策略，这对于提升PSNR指标至关重要；在微调阶段确保ORPO的偏好数据质量，即挑选出的正样本必须明确优于负样本。
*   **迁移可能**：该框架易于迁移至其他需要高成本环境交互的任务（如医疗手术机器人、复杂工具使用）。

### 7. 总结
*   **核心思想**：利用想象中的物理模拟，通过对比学习实现VLM的物理接地。
*   **速记版pipeline**：
    1.  低成本收集少量探索数据。
    2.  训练视频模型预演物体动态。
    3.  挑选模型预测中的最优动作。
    4.  通过对比微调改进规划策略。

**Key Findings:**

- To overcome this bottleneck, we introduce DreamPlan, a novel framework for the reinforcement fine-tuning of VLM planners via video world models.
- We demonstrate that this sub-optimal data is sufficient to train an action-conditioned video generation model, which implicitly captures complex real-world physics.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16860v1)
- [arXiv](https://arxiv.org/abs/2603.16860v1)

---

<a id='2603.16844v1'></a>
## [M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM](https://arxiv.org/abs/2603.16844v1)

**Authors:** Kerui Ren, Guanghao Li, Changjian Jiang, Yingxiang Xu, Tao Lu, Linning Xu, Junting Dong, Jiangmiao Pang, Mulin Yu, Bo Dai

**Published:** 2026-03-17

**Categories:** cs.CV

**Abstract:**

Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization. To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM. M^3 further enhances tracking stability by incorporating dynamic area suppression and cross-inference intrinsic alignment. Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction. Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.

**Analysis:**

### 1. 摘要翻译
单目视频的流式重建仍然具有挑战性，因为它要求在动态环境中同时实现高精度的位姿估计和计算高效的在线细化。虽然将3D基础模型与SLAM框架耦合是一种很有前途的范式，但仍存在一个关键瓶颈：大多数多视图基础模型以“前馈”（feed-forward）方式估计位姿，产生的像素级对应关系缺乏严格几何优化所需的精度。为了解决这一问题，我们提出了**M³**，它通过增加一个专用的**匹配头（Matching head）**来增强多视图基础模型，以促进细粒度的密集对应，并将其集成到鲁棒的单目高斯溅射（Gaussian Splatting）SLAM中。此外，M³通过引入动态区域抑制和跨推理内在对齐（cross-inference intrinsic alignment），进一步增强了跟踪稳定性。在多种室内和室外基准测试上的广泛实验证明，M³在位姿估计和3D重建方面均达到了最先进的精度。

---

### 2. 方法动机分析
*   **驱动力**：解决现有流式重建方法在“高精度几何约束”与“实时计算效率”之间的矛盾。
*   **痛点**：当前基于前馈的多视图基础模型（如Pi3X）虽然能提供位姿，但缺乏像素级的稠密匹配精度，导致SLAM后端的束调整（Bundle Adjustment）无法建立稳健的几何约束，引发鬼影或轨迹漂移。
*   **研究假设**：通过在基础模型中加入专门优化的密集匹配头，并结合SLAM中的位姿先验进行粗到细（Coarse-to-fine）的匹配，可以实现鲁棒且高精度的几何优化，从而提升重建质量。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **增强特征提取**：在Pi3X基础上引入Matching Head，利用DPTdesc模块和MLP输出稠密的特征描述子和匹配置信度图。
    2.  **位姿引导的密集匹配**：利用已估计的位姿将点图映射到当前帧，限制匹配搜索范围，将二次方全局匹配降维至线性局部细化。
    3.  **动态区域估计**：通过描述子相似度与运动图（Motion map）对比，抑制动态物体，防止其引入伪影。
    4.  **端到端优化**：将匹配、动态过滤和高斯溅射重建集成至单次推理中，通过Factor Graph进行全局Bundle Adjustment。
*   **模型结构**：Pi3X Encoder/Decoder + Matching Head（核心新增）+ Gaussian Mapper。
*   **关键公式**：论文通过对称的InfoNCE损失函数（公式3）优化描述子一致性，并利用位姿引导的局部窗口细化搜索（公式4）来提升匹配精度。

---

### 4. 方法对比分析
*   **本质区别**：不再依赖黑盒前馈预测，而是通过Fine-tune出的匹配头，建立像素级几何关联，显式地为SLAM后端提供强几何约束。
*   **创新贡献**：将“匹配”作为模型内置能力而非外部插件，并实现了单次推理下同时完成追踪与建图，解决了冗余计算问题。
*   **适用场景**：适用于单目长视频、大尺度、包含复杂动态物体的真实场景重建。

---

### 5. 实验分析
*   **验证方法**：在ScanNet++、Waymo、KITTI等主流基准上，与DROID-SLAM、ARTDECO等对比。
*   **关键结论**：在ScanNet++上，ATE RMSE精度较VGGT-SLAM 2.0提升了64.3%，在PSNR上优于ARTDECO 2.11dB。
*   **主要优势**：高位姿精度、强鲁棒性（动态抑制）、计算高效。
*   **局限**：严重依赖基础模型的前馈预测质量；目前缺乏多传感器融合能力，对极端不准确的先验缺乏fallback机制。

---

### 6. 实用指南
*   **开源情况**：项目主页：https://city-super.github.io/M3/。
*   **实现细节**：建议关注匹配搜索半径 $r=4$ 的设置；匹配头Fine-tune过程需要使用DINOv2 backbone提取特征。
*   **迁移可能**：匹配头的设计可直接迁移至其他基于多视图几何的任务，如三维语义分割或动态环境追踪。

---

### 7. 总结
*   **核心思想**：通过定制匹配头实现像素级几何匹配，赋能单目流式重建。
*   **速记版pipeline**：
    1. 基础模型输出特征图；
    2. 位姿预测引导局部特征匹配；
    3. 动态区域滤波剔除无效点；
    4. 结合高斯模型进行全局优化。

**Key Findings:**

- To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM.
- Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction.
- Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16844v1)
- [arXiv](https://arxiv.org/abs/2603.16844v1)

---

<a id='2603.16816v1'></a>
## [WildDepth: A Multimodal Dataset for 3D Wildlife Perception and Depth Estimation](https://arxiv.org/abs/2603.16816v1)

**Authors:** Muhammad Aamir, Naoya Muramatsu, Sangyun Shin, Matthew Wijers, Jiaxing Jhong, Xinyu Hou, Amir Patel, Andrew Markham

**Published:** 2026-03-17

**Categories:** cs.CV, cs.DL

**Abstract:**

Depth estimation and 3D reconstruction have been extensively studied as core topics in computer vision. Starting from rigid objects with relatively simple geometric shapes, such as vehicles, the research has expanded to address general objects, including challenging deformable objects, such as humans and animals. However, for the animal, in particular, the majority of existing models are trained based on datasets without metric scale, which can help validate image-only models. To address this limitation, we present WildDepth, a multimodal dataset and benchmark suite for depth estimation, behavior detection, and 3D reconstruction from diverse categories of animals ranging from domestic to wild environments with synchronized RGB and LiDAR. Experimental results show that the use of multi-modal data improves depth reliability by up to 10% RMSE, while RGB-LiDAR fusion enhances 3D reconstruction fidelity by 12% in Chamfer distance. By releasing WildDepth and its benchmarks, we aim to foster robust multimodal perception systems that generalize across domains.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对《WildDepth: A Multimodal Dataset for 3D Wildlife Perception and Depth Estimation》这篇论文的分析如下：

### 1. 论文主要贡献摘要
该论文推出了 **WildDepth** 数据集，这是首个专注于自然界动物的、包含同步 RGB 图像与 LiDAR 点云的多模态数据集。该研究通过引入度量尺度（metric scale）基准，填补了当前动物 3D 感知领域缺乏真实度量数据的空白，显著提升了动物深度估计、行为检测及 3D 重建的精度。

### 2. 关键创新与方法论
*   **多模态融合（RGB-LiDAR）**：不同于以往仅依赖单目 RGB 的深度估计模型，该研究利用 LiDAR 提供精确的几何约束，解决了“无度量尺度”导致的深度模糊问题。
*   **跨环境覆盖**：数据涵盖了从家养到野外的多样化场景，这要求模型具备更强的泛化能力。
*   **基准套件（Benchmark Suite）**：不仅发布了数据集，还提供了包含深度估计、行为检测和 3D 重建的完整评估流程，为该细分领域的算法评估提供了统一标准。

### 3. 对领域的潜在影响
*   **打破“非刚性物体”感知的瓶颈**：动物属于复杂的非刚性形变物体，此数据集将推动深度学习模型从处理“刚性物体”（如自动驾驶中的车辆）向处理“复杂动态生物”进化。
*   **推动度量感知（Metric Perception）的发展**：在缺乏真实几何尺度的情况下，许多计算机视觉模型表现不佳。WildDepth 提供的度量数据将促使学术界开发更具物理真实性的感知算法。
*   **数据驱动的生态学研究**：该工作为计算生态学（Computational Ecology）提供了底层支持，使得利用视觉手段进行野生动物种群监测、行为分析变得更加精确且科学。

### 4. 相关领域或应用受益
*   **野生动物保护与生物学研究**：利用计算机视觉实现非侵入式的自动化动物监测和行为分析。
*   **机器人与自动驾驶**：增强机器人在复杂野外环境下的避障与环境理解能力，特别是应对突发出现的动态生物目标。
*   **电影与数字娱乐**：通过高质量的 3D 重建数据，辅助影视工业中的动物角色动作捕捉与 3D 资产生成。
*   **视觉传感器融合算法开发**：为多模态传感器融合（Sensor Fusion）算法研究者提供了一个极具挑战性的实地测试场。

### 5. 可推测的局限性
*   **数据集规模与多样性局限**：尽管涵盖了多种环境，但“野外”环境的极端光照、天气和遮挡情况可能导致 LiDAR 与 RGB 的对齐（Registration）面临巨大挑战。
*   **LiDAR 传感器的便携性**：野外数据的获取通常需要手持式或移动式激光雷达，这对数据的采集成本和部署设备提出了较高要求，可能限制了大规模数据的进一步扩充。
*   **动态形变下的稀疏性问题**：动物的快速运动可能导致同步过程中产生运动模糊或点云伪影，如何处理极端动态场景下的多模态融合仍然是一个技术难点。

**专家总结：**
这篇论文的意义在于它**将计算机视觉的重点从室内或结构化环境向高复杂度的自然环境进行了实质性的拓展**。它不仅是一个数据集，更是一个通过引入“度量尺度”来校准模型感知能力的重要契机。对于从事视觉感知、多传感器融合及人工智能野生动物保护研究的学者来说，该工作具有重要的参考价值。

**Key Findings:**

- To address this limitation, we present WildDepth, a multimodal dataset and benchmark suite for depth estimation, behavior detection, and 3D reconstruction from diverse categories of animals ranging from domestic to wild environments with synchronized RGB and LiDAR.
- Experimental results show that the use of multi-modal data improves depth reliability by up to 10% RMSE, while RGB-LiDAR fusion enhances 3D reconstruction fidelity by 12% in Chamfer distance.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16816v1)
- [arXiv](https://arxiv.org/abs/2603.16816v1)

---

<a id='2603.16806v1'></a>
## [DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping](https://arxiv.org/abs/2603.16806v1)

**Authors:** Yuliang Wu, Yanhan Lin, WengKit Lao, Yuhao Lin, Yi-Lin Wei, Wei-Shi Zheng, Ancong Wu

**Published:** 2026-03-17

**Categories:** cs.RO, cs.AI

**Abstract:**

To meet the demands of increasingly diverse dexterous hand hardware, it is crucial to develop a policy that enables zero-shot cross-embodiment grasping without redundant re-learning. Cross-embodiment alignment is challenging due to heterogeneous hand kinematics and physical constraints. Existing approaches typically predict intermediate motion targets and retarget them to each embodiment, which may introduce errors and violate embodiment-specific limits, hindering transfer across diverse hands. To overcome these limitations, we propose \textit{DexGrasp-Zero}, a policy that learns universal grasping skills from diverse embodiments, enabling zero-shot transfer to unseen hands. We first introduce a morphology-aligned graph representation that maps each hand's kinematic keypoints to anatomically grounded nodes and equips each node with tri-axial orthogonal motion primitives, enabling structural and semantic alignment across different morphologies. Relying on this graph-based representation, we design a \textit{Morphology-Aligned Graph Convolutional Network} (MAGCN) to encode the graph for policy learning. MAGCN incorporates a \textit{Physical Property Injection} mechanism that fuses hand-specific physical constraints into the graph features, enabling adaptive compensation for varying link lengths and actuation limits for precise and stable grasping. Our extensive simulation evaluations on the YCB dataset demonstrate that our policy, jointly trained on four heterogeneous hands (Allegro, Shadow, Schunk, Ability), achieves an 85\% zero-shot success rate on unseen hardware (LEAP, Inspire), outperforming the state-of-the-art method by 59.5\%. Real-world experiments further evaluate our policy on three robot platforms (LEAP, Inspire, Revo2), achieving an 82\% average success rate on unseen objects.

**Analysis:**

## 1. 摘要翻译
为了满足日益多样化的灵巧手硬件需求，开发一种无需冗余重训练即可实现零样本跨形态灵巧抓取策略至关重要。由于异构手的运动学结构与物理约束各异，跨形态对齐极具挑战。现有方法通常预测中间动作目标并进行重定向，这往往引入误差并违反特定形态的物理限制，阻碍了在不同手上的迁移。为克服这些局限，我们提出了 **DexGrasp-Zero**，一种从多种形态中学习通用抓取技能的策略，实现了对未见过的灵巧手的零样本迁移。我们首先引入了一种形态对齐的图表示，将手的运动学关键点映射为解剖学上的基准节点，并为每个节点配备三轴正交运动基元，从而实现了跨不同形态的结构与语义对齐。基于此图表示，我们设计了形态对齐图卷积网络（MAGCN）来编码图特征进行策略学习。MAGCN结合了物理属性注入机制，将特定手的物理约束融入图特征中，实现了对不同连杆长度和驱动限制的自适应补偿，从而保证了抓取的精准与稳定。在YCB数据集上的广泛仿真实验表明，我们的策略在四种异构手（Allegro, Shadow, Schunk, Ability）上联合训练后，在未见过的硬件（LEAP, Inspire）上实现了85%的零样本成功率，超越了当前最先进方法59.5%。真实环境实验进一步验证了我们的策略在三种机器人平台上的有效性，在未见对象上实现了82%的平均成功率。

## 2. 方法动机分析
*   **驱动力**：旨在摆脱对特定机器人手形态的依赖，解决传统强化学习策略在面对新形态手时需要昂贵的重训练和数据采集的问题。
*   **现有方法痛点**：现有研究通常使用中间动作表示（如指尖位置），通过重定向模块转换为物理指令。这种解耦方式存在“语义鸿沟”，极易生成目标手运动学上不可行的动作，且无法直接利用物理先验。
*   **研究假设**：尽管手部形态迥异，但其运动学结构均可抽象为共享的解剖学功能图（节点），且跨形态的动作控制语义可以通过统一的运动基元空间来对齐。

## 3. 方法设计详解
*   **流程总结**：
    1.  **形态对齐图构建**：基于URDF解析，将手抽象为包含指尖、远节、中节、近节、掌骨和腕部节点的图结构，并建立物理属性图（包含连杆长、关节限位、驱动力限制等）。
    2.  **物理属性注入**：在MAGCN的每一层中，通过物理属性编码器将上述URDF物理先验嵌入到图卷积网络中，使得策略能感知当前手的动力学特性。
    3.  **通用动作基元空间**：定义三种正交基元（Flexion, Abduction, Axial Rotation）以及腕部6-DoF位姿。此空间是“手无关”的，统一了控制语义。
    4.  **固定映射输出**：通过一个预定义的、手特定的映射矩阵 $M_h$（基于系统辨识），直接将通用基元输出转化为各关节的物理位移指令 $\Delta q^h$，彻底避免了训练重定向器。
*   **模型结构**：MAGCN作为核心，通过图卷积层处理手部状态，并利用激活掩码（Activation Mask）屏蔽当前手无法实现的动作基元，结合可行性惩罚项，确保生成的动作是物理有效的。

## 4. 方法对比分析
*   **本质区别**：从“间接重定向”转变为“直接端到端映射+物理先验注入”。它不试图让所有手看起来一样，而是让策略通过图结构理解不同手之间的“功能共性”。
*   **创新贡献**：提出将URDF物理先验显式融入图卷积层的方法，以及将手部动作分解为生物启发式的“运动基元”并映射到物理执行空间，显著提升了泛化性能。
*   **适用场景**：适用于任何具有URDF模型定义的灵巧手硬件，且尤其擅长在缺乏特定手训练数据情况下的零样本部署。

## 5. 实验分析
*   **验证方法**：在仿真环境下（RaiSim），采用4手训练、2手测试的CrossDex协议，并在真实Kinova/Piper臂上进行实物部署。
*   **关键结果**：在零样本迁移下，成功率从基准的26.5%提升至85.0%，且在实物实验中实现了82%的高成功率。
*   **主要优势**：极强的形态泛化能力，训练一次即可推广到未知手，无需 finetuning。
*   **主要局限**：对极小物体抓取时，可能因缺乏触觉反馈导致偶尔的“空抓”；对非 anthropomorphic（类人）构型手存在一定挑战，尽管论文证实了其在Barrett手上的可用性。

## 6. 实用指南
*   **开源情况**：已开源，代码见 https://github.com/YliangWu/DexGrasp-Zero。
*   **实现细节**：
    *   $M_h$ 映射矩阵的构建通过单位激励实验获得，这是最关键的预处理步骤。
    *   在真实环境部署中，利用SAM2分割对象，并使用教师-学生蒸馏策略处理视觉-触觉缺失问题。
*   **迁移可能**：该方法中“以语义节点为核心的图表示”和“固定物理映射”思路完全可以迁移到四足机器人行走、柔性机械臂控制等其他多形态机器人控制任务中。

## 7. 总结
*   **核心思想**：通过解剖学功能图与物理先验注入，实现跨形态的通用灵巧抓取控制。
*   **速记版pipeline**：
    1. 将手转化为功能节点图（URDF解析）；
    2. 将物理属性注入图卷积网络学习；
    3. 输出统一语义的运动基元；
    4. 通过固定规则直接驱动各形态关节。

**Key Findings:**

- To overcome these limitations, we propose \textit{DexGrasp-Zero}, a policy that learns universal grasping skills from diverse embodiments, enabling zero-shot transfer to unseen hands.
- Our extensive simulation evaluations on the YCB dataset demonstrate that our policy, jointly trained on four heterogeneous hands (Allegro, Shadow, Schunk, Ability), achieves an 85\% zero-shot success rate on unseen hardware (LEAP, Inspire), outperforming the state-of-the-art method by 59.5\%.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.16806v1)
- [arXiv](https://arxiv.org/abs/2603.16806v1)

---

