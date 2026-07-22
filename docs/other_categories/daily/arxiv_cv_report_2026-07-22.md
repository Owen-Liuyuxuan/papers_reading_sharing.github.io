time: 20260722

# Arxiv Computer Vision Papers - 2026-07-22

## Executive Summary

# 每日 Arxiv 计算机视觉论文执行摘要（2026-07-21）

## 一、主要主题与趋势

本期10篇论文呈现三大核心趋势：

1. **世界模型与智能体驱动的物理仿真**（第1、4、5、10篇）成为最密集的研究方向。研究者正从静态视觉理解向**可交互、可推理、可引导**的世界建模演进，强调动作建模、物理一致性及认知规划。
2. **3D/4D感知向高效、协作、几何基础化发展**（第3、6、8、9篇），包括流式4D实例建模、参数高效点云适应、黎曼几何约束的3D基础模型，以及多智能体协作检测。
3. **隐私保护与多模态推理**（第2、7篇）开始与核心视觉任务深度融合，分别涉及同态加密下的感知融合和长音频-视频原生工具推理。

## 二、显著创新论文

- **《Agentic Real2Sim》（第5篇）**：首次将视觉语言智能体（VLM agents）引入基于物理的世界模型构建，实现从真实场景到仿真环境的自动化映射，为具身智能与仿真训练提供了新范式。
- **《OmniReasoner》（第2篇）**：提出“原生工具使用”方式处理长音频-视频推理，突破了传统模型对固定时间窗口的依赖，展示了向通用多模态推理演进的潜力。
- **《Latent Riemannian Flow Matching》（第8篇）**：将黎曼流匹配引入潜在空间，为3D基础模型提供几何严格的先验，有望提升3D生成与理解的鲁棒性和泛化性。
- **《WorldScape Policy 2.0》（第10篇）**：结合推理增强记忆实现可引导的世界动作建模，使模型不仅能预测未来，还能根据高层语义指令调整行为，向“可控世界模型”迈出重要一步。

## 三、新兴研究方向

1. **4D流式实例建模**（IGGT4D）：将3D实例分割扩展至时间维度，支持流式处理，对自动驾驶、AR/VR实时场景理解有直接价值。
2. **参数高效的3D点云层次适应**（Point Ladder Tuning）：借鉴Prompt Tuning思想，提出分层调参策略，在保持预训练能力的同时显著降低3D模型微调成本。
3. **多厂商隐私保护感知融合**（Sarus）：首次系统评估同态加密在真实多源感知融合中的性能，为自动驾驶等场景中的隐私合规提供了可行方案。
4. **认知双过程的自动驾驶规划**（第4篇）：引入“系统1直觉推理+系统2逻辑验证”框架，结合结构化场景知识，使规划结果具备可验证的推理-动作一致性。

## 四、建议全文精读的论文

- 若关注**世界模型与智能体前沿**：优先阅读第5篇（Agentic Real2Sim）和第10篇（WorldScape Policy 2.0），了解当前该领域最具突破性的方法论。
- 若关注**3D基础模型与几何学习**：第8篇（Latent Riemannian Flow Matching）提供了新颖的理论框架，值得深入。
- 若关注**实际系统部署中的隐私与效率**：第7篇（Sarus）与第6篇（Point Ladder Tuning）分别从安全和计算角度给出了实用解决方案。
- 若关注**自动驾驶规划与推理**：第4篇（Cognitive Dual-Process Planning）提出的可验证推理机制具有启发性，可结合第3篇（IGGT4D）的4D感知阅读。

---

## Table of Contents

1. [Masked Visual Actions for Unified World Modeling](#2607.19343v1)
2. [OmniReasoner: Thinking with Long Audio-Video via Native Tool Use](#2607.19339v1)
3. [IGGT4D: Streaming 4D Instance-Grounded Geometry Transformer](#2607.19228v1)
4. [Cognitive Dual-Process Planning for Autonomous Driving with Structured Scene Knowledge and Verifiable Reasoning-Action Consistency](#2607.19194v1)
5. [Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents](#2607.19190v1)
6. [Point Ladder Tuning: Parameter-Efficient Hierarchical Adaptation for 3D Point Cloud Understanding](#2607.19171v1)
7. [Sarus: Privacy-Preserving Multi-Vendor Perception Fusion via Homomorphic Encryption](#2607.19146v1)
8. [Latent Riemannian Flow Matching for Geometry-Grounded 3D Foundation Models](#2607.19120v1)
9. [CoGoal3D: Collaborative 3D Object Detection with 3D-Aware Fusion and Refinement](#2607.19036v1)
10. [WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory](#2607.18840v1)

---

## Papers

<a id='2607.19343v1'></a>
## [Masked Visual Actions for Unified World Modeling](https://arxiv.org/abs/2607.19343v1)

**Authors:** Hadi Alzayer, Wenlong Huang, Haonan Chen, Christopher Luey, Lvmin Zhang, Maneesh Agrawala, Gordon Wetzstein, Li Fei-Fei, Yilun Du, Jiajun Wu, Jia-Bin Huang

**Published:** 2026-07-21

**Categories:** cs.CV, cs.RO

**Abstract:**

Video models absorb rich priors over how the visual world moves, interacts, and responds to contact, making them promising substrates for robotic world modeling. The central challenge is how to communicate action to such models in a form aligned with the visual space in which they learned these interaction priors, yet still grounded in physical manipulation. We introduce Masked Visual Actions, a pixel-space control interface that expresses action as a partially revealed trajectory of an arbitrary entity in a video. Revealing robot motion makes the model act as a forward dynamics model that predicts the scene's response to low-level robot actions, while revealing desired object motion makes the same model recover robot behavior consistent with that outcome. Finetuned with only 15 hours of masked examples from real videos and simulation, a single checkpoint achieves strong visual fidelity and controllability across diverse scenes and multiple embodiments. In downstream manipulation settings, the model produces imagined rollouts whose outcomes correlate with real-world execution for policy evaluation, improves decision making by ranking candidate futures in model-based planning, and supports inverse modeling by synthesizing robot motion from desired object motion.

**Analysis:**

作为计算机视觉和机器人领域的专家，我为您分析这篇题为《Masked Visual Actions for Unified World Modeling》的论文。该研究在视频生成模型与具身智能（Embodied AI）的结合上迈出了重要一步。

### 1. 核心贡献总结
该论文提出了一种名为“掩码视觉动作”（Masked Visual Actions, MVA）的统一控制接口，通过在视频像素空间中通过“掩码（Masking）”和“揭示（Revealing）”轨迹来表征动作。这一方法使得单一的生成模型能够同时实现正向动力学预测（给定机器人动作预测环境反馈）和逆向控制求解（给定目标物体运动反推机器人轨迹），从而在无需庞大训练集的情况下，极大地提升了机器人对复杂视觉世界的建模与控制能力。

### 2. 关键创新点与方法论
*   **像素空间控制范式**：不同于传统的以状态向量（State-space）或动作指令（Action space）为核心的控制方式，MVA 直接在视频的视觉特征空间中进行操作。通过将机器人的动作转化为视频中像素的轨迹，模型能够利用预训练视频模型（Video Models）中已习得的物理常识和运动先验。
*   **双向任务统一**：该方法巧妙地将“正向动态模型”（Forward Dynamics）和“逆向动态模型”（Inverse Dynamics/Planning）统一在同一框架内。通过掩码机制，模型可以根据输入的“动作轨迹”预测未来视觉场景，也可以根据输入的“目标物体轨迹”反向引导机器人动作生成。
*   **高效的数据利用**：仅需 15 小时的掩码视频数据进行微调，模型即可表现出跨场景、跨形态（Embodiment）的泛化能力，证明了预训练视觉表征在机器人控制中的高效迁移潜力。

### 3. 对该领域的潜在影响
*   **打破“动作”与“视觉”的壁垒**：该工作展示了动作不必被硬性编码为特定的控制向量，而是可以作为一种视觉特征，这为“视觉即控制（Vision-as-Control）”开辟了新的可能性。
*   **模型驱动的决策优化（Model-based Planning）**：通过“想象（Imagination）”能力，机器人可以在执行前模拟多种未来可能性并进行排名，这显著提升了机器人应对未知环境时的决策质量。
*   **通用具身基础模型的发展**：为构建通用的“物理世界模拟器”提供了可行的技术路线，有望减少对特定机器人本体及其仿真环境的深度依赖。

### 4. 受益的相关领域与应用
*   **机器人操作（Manipulation）**：特别是针对杂乱场景下的复杂抓取与操作任务。
*   **自动驾驶**：用于预测周围交通参与者的行为，并结合本车动作进行轨迹规划。
*   **交互式内容创作与模拟**：通过描述物体移动轨迹来生成相应的物理交互视频。
*   **少样本学习（Few-shot Learning）**：在机器人训练数据稀缺的情况下，利用大规模视频预训练模型进行快速适配。

### 5. 潜在局限性（从摘要推断）
*   **实时计算开销**：作为一种基于视频生成模型的方法，其推理速度可能限制了在极高频动态控制环境中的直接应用，特别是如果模型需要在推理时进行多路径的“想象（Imagination）”。
*   **长程物理一致性**：尽管模型表现出强视觉保真度，但在长序列预测中，视频生成模型普遍存在“漂移”或物理规律违反（如物体凭空消失或穿模）的现象。
*   **细粒度控制精度**：像素空间的控制可能难以满足对力控（Force Control）或极高精度运动轨迹的要求，可能需要与低层控制器（Low-level controller）配合使用。
*   **对掩码策略的依赖**：视觉动作的有效性可能高度依赖于掩码的覆盖范围和时空平滑度，复杂交互（如摩擦、多接触点）下的建模效果有待进一步验证。

**总结：** 这篇论文的趣味性在于它将“动作生成”本质上转化为了一种“填空（In-painting/Completion）”任务。如果该方法能够通过后续研究解决推理延迟问题，它将成为连接大规模预训练视觉模型与物理世界交互的重要桥梁。

**Key Findings:**

- We introduce Masked Visual Actions, a pixel-space control interface that expresses action as a partially revealed trajectory of an arbitrary entity in a video.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19343v1)
- [arXiv](https://arxiv.org/abs/2607.19343v1)

---

<a id='2607.19339v1'></a>
## [OmniReasoner: Thinking with Long Audio-Video via Native Tool Use](https://arxiv.org/abs/2607.19339v1)

**Authors:** Yu Chen, Caorui Li, Ziyu Xiong, Yidong Wang, Mingqi Gao, Shuman Liu, Biao Liu, Chunfeng Yang, Anxiang Zeng, Haibo Zhang, Chaofan Chen

**Published:** 2026-07-21

**Categories:** cs.CV

**Abstract:**

Long audio-video reasoning is difficult for omnimodal LLMs because the decisive evidence is often sparse, cross-modal, and too expensive to preserve with uniformly high-fidelity inputs. We introduce OmniReasoner, a tool-use post-training framework for Thinking with Long Audio-Video: omni-modal LLMs learn, via supervised fine-tuning and reinforcement learning, to decide whether and where to call a zoom-in tool before answering. OmniReasoner first builds a low-cost global preview of the full stream and then, when needed, calls the zoom-in tool with a requested temporal interval for higher-fidelity visual and audio inspection before answering. Because the model observes different sampling granularities before and after this call -- a sparse global preview and a denser local clip -- we introduce TimeAnchor, which keeps the tool's temporal argument valid and round-trip-consistent across these granularities, rather than tied to frame indices from a particular sampling rate. To make this tool-use behavior trainable without expensive manual interval annotation, we build a Temporal Augmented Data Engine that synthesizes tool-use post-training trajectories by video editing and composition. Experiments across omnimodal and video benchmarks show that OmniReasoner improves both answer accuracy and temporal grounding while concentrating high-fidelity computation on informative regions. Code is available at https://github.com/RockyChen0205/OmniReasoner.

**Analysis:**

### 1. 摘要翻译
长音视频推理对全模态大语言模型（LLMs）极具挑战性，因为关键证据往往稀疏、跨模态且在全长视频中保持高保真度输入成本过高。我们引入了 **OmniReasoner**，一个用于长音视频“思考”的工具使用后训练框架。通过监督微调（SFT）和强化学习（RL），全模态LLM能够学习在回答前决定是否以及在何处调用缩放（zoom-in）工具。OmniReasoner首先构建全长流的低成本全局预览，按需调用缩放工具以获取高保真视频和音频片段进行检查。由于模型在工具调用前后观察到的采样粒度不同（稀疏全局预览 vs. 稠密局部片段），我们引入了 **TimeAnchor**，它确保工具的时间参数在不同粒度间保持有效且往返一致，而非依赖于特定采样率的帧索引。为使这种工具使用行为在无需昂贵手动标注的情况下可训练，我们构建了“时间增强数据引擎”，通过视频编辑和组合合成工具使用后的训练轨迹。实验表明，OmniReasoner在提高回答准确性和时间定位能力的同时，将高保真计算集中于信息密集区域。代码已开源。

### 2. 方法动机分析
*   **驱动力**：解决长视频（长达一小时）理解中“全覆盖”与“高保真”的矛盾，实现类似人类在长文档中“快速浏览-定位怀疑点-深入观察”的推理模式。
*   **痛点**：现有方法大多采取均匀采样，导致长视频中的细粒度关键证据（如关键音、微动作）在下采样中丢失，或因处理全长高分辨率视频而面临严重的计算冗余。
*   **研究假设**：通过引入“缩放工具”和“绝对时间锚点（TimeAnchor）”，模型能够学会动态分配计算资源，在稀疏预览与密集局部检查之间建立稳定的时间语义关联，从而提升长视频推理性能。

### 3. 方法设计详解
*   **流程 pipeline**：
    1.  **全局预览 (Global Observation)**：模型首先处理低帧率视觉和完整音频，生成稀疏全局上下文。
    2.  **工具决策 (Tool Decision)**：根据全局上下文与问题，模型决定直接回答或调用缩放工具。
    3.  **时间锚定 (TimeAnchor)**：在令牌序列前端预置绝对时间戳（如 `<t s>`），使缩放工具调用的时间参数在任何粒度下都对应同一物理时刻。
    4.  **局部检查 (Local Evidence)**：系统根据模型请求的 `[s, e]` 时间窗口，从视频源中截取高保真片段，模型结合全局背景与局部细节生成最终答案。
*   **核心模块**：
    *   **TimeAnchor**：通过在每个2秒数据块前添加绝对文本时间戳，消除了帧索引随采样率变化的歧义。
    *   **Temporal Augmented Data Engine**：通过合成技术（多片段拼接、异常插入），自动生成包含“全局预览-工具调用-局部证据-答案”的完整训练轨迹。

### 4. 方法对比分析
*   **本质区别**：与现有agentic系统（仅处理视觉）不同，OmniReasoner是原生的全模态工具使用框架，不仅关注“在哪里看”，更关注“在哪里听”以及“跨模态证据的瞬时一致性”。
*   **创新贡献**：TimeAnchor 机制巧妙地解决了多粒度采样下的时间一致性难题，且数据引擎无需昂贵的人工标注。
*   **适用场景**：需要处理长视频中稀疏线索的复杂问答（如长视频中的计数、因果推理、异常检测）。

### 5. 实验分析（精简版）
*   **关键结论**：在OmniVideoBench、LVOmniBench等多项长视频基准测试中大幅超越基线，视频越长（10-30分钟），提升效果越明显（提升9.9个点）。
*   **主要优势**：不仅提升了准确率，还通过动态缩放显著优化了计算效率。
*   **主要局限**：模型对工具的使用逻辑依赖于SFT的预训练引导；当前工具仅支持缩放（zoom-in），未涵盖外部搜索等更复杂的Agent行为。

### 6. 实用指南
*   **实现细节**：
    *   **RL阶段**：使用GRPO算法进行策略优化，奖励函数 $R = R_{acc} + R_{fmt}$ 分别用于准确率检查和格式规范检查。
    *   **数据构建**：必须利用FFmpeg构建 `MediaSandbox`，确保训练时可随时动态截取片段，且需注意过滤掉可以通过文本直接猜出的“弱智”题目。
*   **迁移建议**：TimeAnchor机制可直接迁移到任何依赖视频分块处理的VLM架构中，解决多粒度采样下的时间定位偏移问题。

### 7. 总结
*   **核心思想**：通过绝对时间锚点实现多粒度下的动态视觉缩放推理。
*   **速记版pipeline**：
    1. 快速扫描全片获取全局概览；
    2. 基于概览决定是否需深入检查；
    3. 若需检查，通过时间锚点精准定位片段；
    4. 结合局部细节与全局背景得出答案。

**Key Findings:**

- We introduce OmniReasoner, a tool-use post-training framework for Thinking with Long Audio-Video: omni-modal LLMs learn, via supervised fine-tuning and reinforcement learning, to decide whether and where to call a zoom-in tool before answering.
- Because the model observes different sampling granularities before and after this call -- a sparse global preview and a denser local clip -- we introduce TimeAnchor, which keeps the tool's temporal argument valid and round-trip-consistent across these granularities, rather than tied to frame indices from a particular sampling rate.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19339v1)
- [arXiv](https://arxiv.org/abs/2607.19339v1)

---

<a id='2607.19228v1'></a>
## [IGGT4D: Streaming 4D Instance-Grounded Geometry Transformer](https://arxiv.org/abs/2607.19228v1)

**Authors:** Zhengyu Zou, Hao Li, Kuixuan Jiao, Liu Liu, Tingyang Xiao, Xiaolin Zhou, Fangzhou Hong, Zhizhong Su, Dingwen Zhang, Ziwei Liu

**Published:** 2026-07-21

**Categories:** cs.CV

**Abstract:**

Real-world spatial intelligence requires agents to understand scenes from continuous video streams, where objects move, persist, disappear, and reappear over time. While recent spatial foundation models have enabled generalizable feed-forward 3D reconstruction, most streaming methods remain geometry-centric and lack temporally consistent object-level understanding. Meanwhile, existing semantic reconstruction and 3D-aware vision-language methods largely rely on externally extracted 2D semantic cues or loosely coupled geometry inputs, limiting unified geometry-instance learning in long dynamic scenes. In this paper, we propose IGGT4D, a streaming instance-grounded geometry Transformer for online 4D scene understanding. IGGT4D processes video frames sequentially, reuses historical context through causal spatial-temporal modeling, and incrementally updates a unified representation of camera motion, geometry, and object identity. This enables long-sequence feed-forward reconstruction with geometry-instance consistency in dynamic environments. To address the lack of high-quality 4D supervision, we further construct InsScene4D-147K, a large-scale dataset spanning real/synthetic and static/dynamic scenes, with RGB images, depth, poses, and temporally consistent instance masks generated by an automated geometry-guided annotation pipeline. Experiments on 3D reconstruction, pose estimation, instance spatial tracking, and open-vocabulary segmentation demonstrate that IGGT4D outperforms existing streaming baselines while maintaining scalable online inference for long dynamic sequences.

**Analysis:**

这是一份关于 **IGGT4D: Streaming 4D Instance-Grounded Geometry Transformer** 的深度分析报告。

---

### 1. 摘要翻译
真实世界的空间智能要求智能体能够理解持续视频流中的场景，即物体随时间移动、持久存在、消失及重现。虽然近期的空间基础模型实现了通用的前馈3D重建，但大多数流式方法仍以几何为中心，缺乏时间上一致的对象级理解。同时，现有的语义重建和3D感知视觉-语言方法在很大程度上依赖于外部提取的2D语义线索或松散耦合的几何输入，限制了长动态场景中的统一几何-实例学习。本文提出了 **IGGT4D**，一种用于在线4D场景理解的流式实例接地几何Transformer。IGGT4D逐帧处理视频，通过因果时空建模复用历史上下文，并增量更新摄像机运动、场景几何和物体身份的统一表示。这使得在长序列中实现了具有几何-实例一致性的前馈重建。

### 2. 方法动机分析
*   **驱动力**：将4D场景理解从静态全序列优化问题转化为**实时、在线的因果流式预测问题**。
*   **痛点**：现有流式3D方法（如Spann3R, Stream3R）仅关注几何一致性，缺乏“物体身份”的持久性理解；而依赖外部2D语义的方法在长序列中难以维持物体在空间和时间上的恒定性。
*   **研究假设**：通过在几何重建过程中“以物体身份为一等公民”进行联合训练，并将实例关联接地于3D几何结构，可以实现不需要全序列优化的稳定、一致的4D在线重建。

### 3. 方法设计详解
*   **流程总结**：
    1.  **因果输入与编码**：将图像流配合摄像机参数编码，利用Transformer块实现因果注意力（仅关注当前及过去帧），通过KV缓存复用历史信息。
    2.  **Tri-DPT头**：设计了一个三分支解码器（Depth, Ray, Instance），不仅预测几何，还通过**几何感知注意力（Geo-Attn）**将深度和射线信息作为结构先验注入到实例分支。
    3.  **流式实例聚类**：摒弃了高开销的HDBSCAN，采用轻量级编码器（Instance Codebook），通过逐帧的**余弦相似度匹配**和**面积加权融合**（Area-weighted fusion）在线维护实例中心和ID。
*   **核心逻辑**：将物体特征（Instance Features）与3D点云的几何位置紧密绑定，即使物体被遮挡或移出视野，通过几何约束也能在重现时保持ID不变。

### 4. 方法对比分析
*   **本质区别**：与仅关注几何的流式模型不同，它将“实例一致性”作为预测目标。与离线模型（IGGT）不同，它采用了因果流式推理，解决了内存随时间剧增的问题（OOM）。
*   **创新贡献**：提出了**流式实例接地（Instance-Grounded）框架**，并利用首帧几何归一化（FF-Norm）解决了长流式序列中的尺度漂移问题。
*   **适用场景**：机器人导航、实时三维语义扫描、智能体长时记忆场景理解。

### 5. 实验分析
*   **验证方法**：在HiRoom、ETH3D、7Scenes、ScanNet++及HOI4D等数据集上进行基准测试。
*   **关键结果**：在长序列任务中，相比基线模型，不仅实现了高效的在线推理（GPU内存固定在0.7GB左右），而且在相机位姿估计和3D重建精度上均表现领先。
*   **局限性**：仍处于监督学习阶段，依赖标注数据集的质量；对于极度复杂的未知动态环境，泛化能力仍有待通过自监督学习进一步提升。

### 6. 实用指南
*   **开源情况**：代码和项目主页位于 [iggt4d.github.io](https://iggt4d.github.io)。
*   **实现要点**：必须使用**首帧几何归一化（FF-Norm）**，否则长序列会产生严重的尺度模糊。Tri-DPT头的设计是性能提升的关键，建议复现时严格遵循其多尺度特征融合策略。
*   **迁移建议**：该模块可以作为一个即插即用的“实例感知头”接入任何现有的流式视觉Transformer框架中。

### 7. 总结
*   **核心思想**：通过几何结构约束，实现长视频流中物体身份的实时、在线持续追踪。
*   **速记版Pipeline**：
    1. 逐帧编码并复用历史缓存。
    2. 几何与实例特征联合预测。
    3. 基于几何线索完成在线实例聚类。
    4. 增量更新全局ID表。

**Key Findings:**

- In this paper, we propose IGGT4D, a streaming instance-grounded geometry Transformer for online 4D scene understanding.
- Experiments on 3D reconstruction, pose estimation, instance spatial tracking, and open-vocabulary segmentation demonstrate that IGGT4D outperforms existing streaming baselines while maintaining scalable online inference for long dynamic sequences.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19228v1)
- [arXiv](https://arxiv.org/abs/2607.19228v1)

---

<a id='2607.19194v1'></a>
## [Cognitive Dual-Process Planning for Autonomous Driving with Structured Scene Knowledge and Verifiable Reasoning-Action Consistency](https://arxiv.org/abs/2607.19194v1)

**Authors:** Zhongyao Yang, Haoyu Li, Yu Yan, Zhuangxuan Yu, Jiangfeng Nan, Jinrui Nan

**Published:** 2026-07-21

**Categories:** cs.RO, cs.CV

**Abstract:**

High-level planning for autonomous driving is a knowledge-intensive engineering decision task that requires accurate scene understanding, timely inference, and internally consistent action selection. Vision-language models (VLMs) can make intermediate reasoning explicit, but their use in deployed planners is constrained by costly structured supervision, unnecessary reasoning in routine scenes, and possible inconsistencies between generated rationales and driving actions. We present a cognitive dual-process planning framework that represents planning-relevant scene knowledge in a machine-parsable structured chain-of-thought (S-CoT) schema. An automated data engine integrates perception foundation models, critical-path filtering, and an expert VLM to generate S-CoT supervision without manual annotation of individual rationales. A lightweight visual Arbiter estimates scene complexity from multilevel vision-encoder features before language decoding and routes each input to either fast meta-action prediction or slow structured reasoning. For slow-path outputs, a deterministic rule-based validator checks whether the parsed S-CoT fields are consistent with the final meta-action and provides verifiable rewards for Group Relative Policy Optimization (GRPO). In a 195-scene manual audit, the generated annotations achieve 91.8\% CoT accuracy and a 98.5\% Logical Consistency Score (LCS). On 574 manually verified NAVSIM test samples, the planner achieves 80.14\% planning accuracy and 97.20\% LCS while reducing average latency by 17.39\% relative to applying slow reasoning to every scene. Evaluation on external long-tail subsets further identifies conditions under which routing and planning performance degrade. Together, these results show how explicit scene knowledge can be operationalized through adaptive reasoning and rule-based verification to support high-level VLM planning decisions.

**Analysis:**

这是一篇将认知心理学理论引入自动驾驶决策规划的前沿论文。以下是对该论文的深度分析：

### 1. 核心贡献总结
该论文提出了一种**认知双进程规划框架（Cognitive Dual-Process Planning）**，通过引入“结构化思维链（S-CoT）”实现自动驾驶决策的推理与行动一致性。该框架通过一个轻量级“仲裁器（Arbiter）”动态分配任务路径，既保留了大型视觉语言模型（VLM）在复杂场景下的推理能力，又通过自动化数据引擎和规则验证器解决了人工标注成本高及逻辑一致性差的行业痛点。

### 2. 关键创新与方法论
*   **双系统架构（Dual-Process Architecture）：** 借鉴了心理学中的双加工理论，利用“快思考”（元动作预测）处理常规场景，利用“慢思考”（结构化推理）处理复杂场景，显著降低了推理延迟（减少了17.39%）。
*   **结构化思维链（S-CoT）：** 将复杂的自然语言推理转化为机器可解析的结构化格式。这使得推理过程不再是不可控的“黑盒”，而是成为可验证的中间产物。
*   **全流程自动化数据闭环：** 通过融合感知基础模型（Foundation Models）、关键路径过滤和专家VLM，实现了无需人工标注的S-CoT数据生成，大幅提升了数据构建的效率与规模。
*   **逻辑一致性验证器（Verifiable Reasoning-Action Consistency）：** 引入确定性规则验证器，检查S-CoT与最终动作的一致性，并以此作为反馈信号指导GRPO（组相对策略优化），确保规划输出在逻辑上是“诚实”的。

### 3. 对领域的潜在影响
*   **改变“端到端”黑盒特性：** 在深度学习规划模型中引入明确的逻辑一致性约束，有助于解决目前自动驾驶模型中常出现的“动作正确但原因错误（Right for the wrong reasons）”的鲁棒性问题。
*   **资源利用率提升：** 为端到端大模型在自动驾驶车端部署提供了切实可行的路径——通过轻量化路由机制，使得大型推理模型仅在必要时启动，解决了计算成本高昂的制约。
*   **可解释性与安全性结合：** 将可解释推理（Reasoning）直接作为强化学习的监督信号，为构建安全、可信的AI驾驶员提供了技术范式。

### 4. 受益的相关领域与应用
*   **机器人运动规划（Robotic Motion Planning）：** 同样适用于需要复杂逻辑判断的工业机器人或服务机器人任务。
*   **嵌入式AI与边缘计算：** 其“动态路由”思想对所有需要高性能推理但算力受限的边缘设备（如无人机、AR设备）具有借鉴意义。
*   **具身智能（Embodied AI）：** 论文提出的自动化数据引擎和逻辑验证流程，可加速具身智能在复杂环境中的泛化能力。

### 5. 可推断的局限性
*   **长尾场景性能下降：** 摘要提到“在外部长尾子集上的评估显示路由和规划性能下降”，这说明当遇到极端罕见场景（Out-of-distribution）时，轻量级的仲裁器可能无法准确判断是否需要启动“慢推理”，从而导致漏判或误判。
*   **规则依赖性（Rule-based Bottleneck）：** 验证器依赖于规则，如果场景过于复杂或超出了预设的结构化字段定义，规则系统可能成为瓶颈或导致误导性的惩罚。
*   **对VLM基础能力的依赖：** 虽然采用了自动化标注，但系统的上限依然受限于所使用的“专家VLM”的能力。若专家模型无法理解某些复杂边缘场景，则生成的S-CoT数据质量将大打折扣。

### 专家点评
这篇论文的精妙之处在于它不仅提出了一个架构，还**重新定义了“训练数据”的范式**——将原本不可言说的驾驶经验转化为结构化的推理数据。这种将**认知心理学的架构逻辑**与**强化学习策略优化（GRPO）**结合的方法，是当前自动驾驶领域从单纯的“模仿学习”向“推理学习”过渡的重要里程碑。

**Key Findings:**

- We present a cognitive dual-process planning framework that represents planning-relevant scene knowledge in a machine-parsable structured chain-of-thought (S-CoT) schema.
- Together, these results show how explicit scene knowledge can be operationalized through adaptive reasoning and rule-based verification to support high-level VLM planning decisions.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19194v1)
- [arXiv](https://arxiv.org/abs/2607.19194v1)

---

<a id='2607.19190v1'></a>
## [Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents](https://arxiv.org/abs/2607.19190v1)

**Authors:** Guanxiong Chen, Qianjun Xia, Jiawei Peng, Heng Zhang, Bole Ma, Justin Qian, Ziyi Jiao, Bingyang Zhou, Luoxin Ye, Kaifeng Zhang, Kunyi Wang, Weijia Zeng, Yunuo Chen, Pengzhi Yang, Ziqiu Zeng, Huamin Wang, Chao Liu, Alan Yuille, Fan Shi, Changxi Zheng, Yunzhu Li, Chenfanfu Jiang, Peter Yichen Chen

**Published:** 2026-07-21

**Categories:** cs.RO, cs.AI

**Abstract:**

Real-to-sim conversion for robotic interaction with objects remains labor-intensive because it requires more than visual reconstruction: a streamlined real2sim process must recover scene geometries and object states, infer physical parameters, and assemble actors, objects, cameras, poses, and trajectories into a runnable physical simulation. Today this process still depends on manual tuning of visual foundation models, mesh cleanup, coordinate-frame alignment, and brittle workflow glue across visual perception tools and simulators. We introduce \textit{Agentic Real2Sim}, a framework for generalized physical world modeling with vision-language agents, converting a real-world recording of object-robot interaction into a simulatable episodic twin which preserves observations, geometries, robot interactions, and object states. We evaluate Agentic Real2Sim on rigid-object manipulation, deformable-object interaction, and humanoid motion scenes, spanning domains that are usually handled by separate Real2Sim pipelines, marking a first step toward scalable conversion. The framework's agentic decisions can be driven by an open-weight VLM backend at a small fraction of the cost of frontier models, while attaining comparable conversion success rate. We aim to use the resulting real-world-aligned twins for downstream robotics tasks, specifically policy learning and evaluation. The project site is available at https://ericchen321.github.io/agentic_real2sim.github.io/.

**Analysis:**

### 1. 摘要翻译
机器人交互的真实场景到模拟器（Real2Sim）转换依然高度依赖人工，这不仅限于视觉重建，还包括恢复物体状态、推断物理参数，并将动作、物体、摄像机位姿和轨迹组装成可运行的仿真。本文提出了 **Agentic Real2Sim**，这是一个利用视觉语言模型（VLM）智能体进行广义物理世界建模的框架。它能将真实的机器人交互记录转换为包含观测、几何、机器人交互和物体状态的“仿真孪生体”（Simulatable episodic twin）。该框架在刚体操作、可变形物体交互和人形运动场景中表现出跨域的可扩展性。其智能决策由开放权重VLM驱动，在大幅降低成本的同时，达到了与商业模型相当的转换成功率。

---

### 2. 方法动机分析
- **驱动力**：解决机器人领域将真实世界交互视频自动转化为可重现、可仿真的数字孪生难题，从而实现大规模下游策略学习与评估。
- **痛点**：现有流程极度依赖手动调参、Mesh修复、坐标对齐及脆弱的软件衔接。以往工作要么仅侧重资产重建，要么侧重策略训练，缺乏一个通用的、能处理完整交互轨迹的“转换管道”。
- **研究假设**：通过将复杂的物理建模任务拆解为由VLM智能体引导的模块化步骤（分而治之），并引入闭环反馈机制，可以实现复杂交互轨迹的自动转换。

---

### 3. 方法设计详解
#### 流程总结
1. **视觉处理阶段**：通过SAM 3及深度估算工具提取关键帧，VLM智能体负责“发现”相关物体，利用Mask Critic进行多次过滤，确保分割与跟踪质量。
2. **物理先验推断**：VLM根据视觉上下文（如材料、物体属性）推断物体的质量和物理参数，为仿真器提供初始化数据。
3. **场景准备阶段**：通过确定性校准工具对齐机器人基座与地面，并在MuJoCo中加载资产。
4. **仿真循环优化**：引入“Simulator-in-the-loop”机制，通过“抓取扫描（Grasp Sweep）”或LLM引导的重试循环，微调物体位姿，确保仿真器内的抓取行为成功。

#### 模型结构与协作
- **智能体（Agent）层**：负责高层逻辑决策（如选哪个框、是否重试、任务上下文解析）。
- **确定性工具（Tool）层**：负责具体计算（如Mesh重建、6D位姿对齐、物理模拟渲染）。
- **协同逻辑**：这种“智能体+工具”的解耦架构使得更换VLM后端仅需替换决策层，无需重写底层的感知与模拟工具。

---

### 4. 方法对比分析
- **本质区别**：与过去仅做资产生成的模型不同，它是一个“episode-level”的转换器，目标不是单一资产，而是包含时序动作的整个交互过程。
- **创新点**：引入了基于VLM的“评价-修正”循环和统一的“episode contract”（片段契约），使得刚体、可变形物体、人形机器人能复用同一套转换逻辑。
- **适用场景**：大规模机器人数据集（如DROID）的自动仿真化，特别适用于缺乏物理模拟数据但有大量真实操作视频的领域。

---

### 5. 实验分析
- **验证方法**：在DROID-100数据集上，对比不同VLM后端的重演成功率（replay-success）及模型使用成本。
- **关键结果**：使用31B参数的开源模型（Gemma）达到了与顶级闭源模型（GPT-5.4）相当的成功率，但成本仅为后者的3%。
- **优势**：极佳的经济性和跨模态适应能力。
- **局限**：受限于上游感知组件的鲁棒性（如分割错误），目前总体成功率尚不足50%，仍有较大的提升空间。

---

### 6. 实用指南
- **开源情况**：项目主页 `https://agentic-real2sim.github.io/`。
- **迁移建议**：该方法模块化程度极高，若要迁移到新任务，只需定义特定的“工具（Tool）”接口，并修改VLM在场景描述中的Schema约束即可。
- **注意点**：需要关注“Mask Critic”的重试次数限制，过高会增加API调用成本，过低则会导致场景重建失败。

---

### 7. 总结
- **核心思想**：利用VLM智能体 orchestrate 确定性工具链，实现交互视频到物理仿真孪生的自动转换。
- **速记版pipeline**：
    1. **视觉提取**：用模型分割物体，VLM选出关键目标。
    2. **属性建模**：VLM推断物理参数并初始化仿真场景。
    3. **物理优化**：仿真器内循环微调位姿，确保动作逻辑一致。
    4. **评价验收**：基于重演结果，自动判断转换是否成功。

**Key Findings:**

- We introduce \textit{Agentic Real2Sim}, a framework for generalized physical world modeling with vision-language agents, converting a real-world recording of object-robot interaction into a simulatable episodic twin which preserves observations, geometries, robot interactions, and object states.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19190v1)
- [arXiv](https://arxiv.org/abs/2607.19190v1)

---

<a id='2607.19171v1'></a>
## [Point Ladder Tuning: Parameter-Efficient Hierarchical Adaptation for 3D Point Cloud Understanding](https://arxiv.org/abs/2607.19171v1)

**Authors:** Junlin Chang, Longhao Zou, Rui Li

**Published:** 2026-07-21

**Categories:** cs.CV

**Abstract:**

Fine-tuning pre-trained point-cloud backbones typically updates all parameters, resulting in substantial computation and memory overhead. More importantly, modern point backbones rely on aggressive tokenization and downsampling, which yields compact global tokens but irreversibly discards fine-grained local geometry, an inherent bottleneck for parameter-efficient adaptation. Consequently, existing PEFT methods that operate only on these coarsened tokens can modulate global semantics but struggle to recover the missing multi-scale locality. We present Point Ladder Tuning (PLT), a locality-aware PEFT framework that performs hierarchical, instance-conditioned adaptation while keeping the backbone frozen. PLT forms a lightweight closed loop: (i) a Hierarchical Ladder Network (HLN) constructs a multi-resolution local feature pyramid directly from raw points; (ii) a Local-Global Fusion (LGF) aligns and fuses local pyramids with intermediate backbone semantics; and (iii) a Dynamic Prompt Generator produces instance-aware multi-scale prompts to modulate the frozen backbone effectively. For dense prediction, we further introduce a lightweight segmentation head that progressively upsamples fused features and leverages backbone priors to refine fine structures. Extensive experiments on classification and dense prediction show that PLT consistently surpasses prior PEFT baselines with minimal tunable parameters. PLT achieves state-of-the-art performance using only 2.71% trainable parameters for classification and 7.69% for dense prediction, and scales favorably to larger backbones, requiring merely 0.36% parameters on PointGPT-L. The code is released at https://github.com/JunLinChang/ECCV2026-PLT.

**Analysis:**

# Point Ladder Tuning (PLT) 方法分析

### 1. 摘要翻译
微调预训练点云骨干网络通常需更新所有参数，带来巨大的计算和内存开销。更关键的是，现代点云骨干网络依赖激进的标记化（tokenization）和下采样，虽然得到了紧凑的全局标记，却不可逆地丢失了细粒度的局部几何结构，这是参数高效微调（PEFT）的内在瓶颈。因此，现有仅在粗化标记上操作的PEFT方法可以调节全局语义，却难以恢复缺失的多尺度局部性。本文提出了**点梯子微调（Point Ladder Tuning, PLT）**，这是一种位置感知（locality-aware）的PEFT框架，在保持骨干冻结的前提下执行层级化、实例条件的微调。PLT构建了一个轻量级的闭环：(i) 层级梯子网络（HLN）直接从原始点构建多分辨率局部特征金字塔；(ii) 局部-全局融合（LGF）模块将局部金字塔与中间骨干语义对齐并融合；(iii) 动态提示生成器（Dynamic Prompt Generator）产生实例感知的多尺度提示，以有效地调制冻结的骨干网络。对于稠密预测，我们引入了轻量级分割头，通过逐步上采样融合特征并利用骨干先验来细化精细结构。在分类和稠密预测上的广泛实验表明，PLT以极少的参数量持续超越先前的PEFT基线，并在PointGPT-L等大模型上表现出优越的可扩展性。

### 2. 方法动机分析
*   **驱动力**：旨在解决在参数高效微调（PEFT）过程中，预训练点云骨干网由于激进下采样导致细粒度几何信息丢失，进而影响下游任务（尤其是稠密预测）精度的问题。
*   **现有方法痛点**：现有PEFT方法大多仅针对经过严重下采样（如2048降至128）的“粗化”标记进行调节，缺失了点云理解中至关重要的细粒度局部结构。
*   **研究假设**：通过额外引入一个轻量级的层级化分支，从原始点直接提取并向冻结的骨干网络动态注入多尺度局部几何先验，可以有效补充全局语义，提升微调性能。

### 3. 方法设计详解
PLT采用“构建-融合-反馈（construct-fuse-feedback）”原则：
1.  **层级梯子网络 (HLN)**：直接从原始输入点云 $P$ 通过多次集合抽象（SA）操作，构造多分辨率局部特征金字塔 $F_l$。这保留了空间粒度。
2.  **局部-全局融合模块 (LGF)**：这是核心桥梁。它将HLN的局部特征 $T_l$ 与骨干网络冻结的全局特征 $T_g$ 进行对齐。通过计算两者的紧凑描述符，利用Attention机制自适应地分配权重，生成融合后的多尺度表示 $F_o$，实现了局部与全局信息的动态交互。
3.  **动态提示生成 (Dynamic Prompt)**：利用融合特征 $F_o$ 生成实例感知提示 $P$，并将其注入到骨干网络的每一层 Transformer 中，以缩放和位移的方式（Scale-and-Shift）调节中间层特征，实现任务自适应。
4.  **轻量级分割头**：为解决稠密预测，利用HLN的多分辨率结构，通过反向距离加权插值逐步恢复分辨率，并结合骨干先验实现精细化的逐点预测。

### 4. 方法对比分析
*   **本质区别**：不同于传统的仅微调标记或在层间插入适配器，PLT通过独立的局部梯子支路，显式地将原始几何信息融入骨干网络，形成了一种协同的双路径适应架构。
*   **创新点**：(1) **显式局部重建**：利用HLN显式补足了丢失的局部几何信息；(2) **选择性语义融合**：LGF模块不仅是简单拼接，而是通过Attention自适应平衡局部与全局语义；(3) **动态多尺度注入**：将多尺度局部先验以实例感知提示的形式注入骨干网络。

### 5. 实验分析（精简版）
*   **验证方法**：在 ScanObjectNN、ModelNet40（分类）及 ShapeNetPart、S3DIS、ScanNetV2（分割）数据集上，对 Point-BERT、Point-MAE、ACT、PointGPT 等基线模型进行PEFT实验。
*   **结论**：在仅使用不到3%参数的情况下，PLT在各项指标上均优于现有的PEFT方法，特别是在最困难的 PB_T50_RS 数据集上，表现出极强的鲁棒性。

### 6. 实用指南
*   **开源地址**：[https://github.com/JunLinChang/ECCV2026-PLT](https://github.com/JunLinChang/ECCV2026-PLT)
*   **实现要点**：核心在于保持预训练骨干网络权重的完全冻结，只更新 HLN 支路和 Prompt 注入部分的参数。数据增强与原始基线保持一致即可。
*   **迁移建议**：该方法逻辑通用，可直接迁移至任何具有标记（token）化机制的3D Transformer 架构。

### 7. 总结
*   **核心思想**：通过独立的局部梯子分支，向冻结的全局模型注入多尺度几何先验。
*   **速记版pipeline**：
    1.  分支提取：通过HLN从原始点重建几何金字塔。
    2.  动态融合：通过LGF将局部几何与全局先验融合。
    3.  提示注入：生成动态提示调制骨干Transformer。
    4.  分层预测：利用多尺度特征进行逐点上采样。

**Key Findings:**

- We present Point Ladder Tuning (PLT), a locality-aware PEFT framework that performs hierarchical, instance-conditioned adaptation while keeping the backbone frozen.
- PLT achieves state-of-the-art performance using only 2.71% trainable parameters for classification and 7.69% for dense prediction, and scales favorably to larger backbones, requiring merely 0.36% parameters on PointGPT-L.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19171v1)
- [arXiv](https://arxiv.org/abs/2607.19171v1)

---

<a id='2607.19146v1'></a>
## [Sarus: Privacy-Preserving Multi-Vendor Perception Fusion via Homomorphic Encryption](https://arxiv.org/abs/2607.19146v1)

**Authors:** Munawar Hasan, Apostol Vassilev

**Published:** 2026-07-21

**Categories:** cs.CR, cs.CV

**Abstract:**

Cooperative perception enables autonomous vehicles (AVs) to improve situational awareness by aggregating detection outputs from multiple agents and sensing platforms, often via a shared fusion service in multi-vendor deployments. However, sharing such outputs at inference time exposes proprietary model behavior and sensitive environmental information, creating significant privacy and security concerns. In this paper, we present Sarus, a privacy-preserving framework for multi-vendor perception fusion via homomorphic encryption (HE), enabling aggregation without revealing individual vendor outputs. Each vendor encodes detections as compact Gaussian moment vectors over a shared spatial lattice and transmits encrypted payloads to a fusion server, which aggregates them directly in the encrypted domain. The fused result is then decrypted and reconstructed into final detections through class-wise bin merging.   We analyze the computational complexity, showing linear scaling for vendor payload construction and $O(BV)$ server-side fusion with the number of occupied bins $B$ and vendors $V$, while postprocessing scales as $O(B + \sum_{c\in \mathcal{C}} B_c^2)$, where $\mathcal{C}$ denotes the set of object classes and $B_c$ is the number of occupied bins for class $c$. Experiments demonstrate linear scaling in practice with only a bounded constant-factor overhead from HE, with decryption dominating postprocessing cost. Experiments on the KITTI dataset using camera (YOLOv8) and LiDAR (PointPillars, PV-RCNN) detectors show that Sarus improves scene-level coverage by effectively aggregating complementary detections, particularly in distance-dependent regimes where individual modalities degrade. These results indicate that privacy-preserving multi-vendor perception fusion is feasible for real-time deployment when statistical compression and spatial sparsity are jointly exploited.

**Analysis:**

### 1. 摘要翻译
协同感知使自动驾驶车辆能够通过聚合多个代理和传感平台的输出来提高态势感知能力，这通常通过多供应商部署中的共享融合服务来实现。然而，在推理时共享此类输出会暴露专有的模型行为和敏感的环境信息，从而带来重大的隐私和安全隐患。在本文中，我们提出了 **Sarus**，这是一个用于通过同态加密（HE）进行多供应商感知融合的隐私保护框架，它能在不泄露各供应商具体输出的情况下实现聚合。每个供应商将检测结果编码为共享空间网格上的紧凑高斯矩向量，并将加密载荷传输到融合服务器，服务器直接在加密域中进行聚合。融合结果随后被解密，并通过类级分箱合并重建为最终检测结果。实验表明，该方法在实际应用中呈线性扩展，仅具有有限的常数级HE开销，并在KITTI数据集上通过有效聚合互补检测，显著提高了场景级覆盖率。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决多供应商协同感知系统中“效用与隐私”的根本矛盾：如何在不暴露供应商模型细节和敏感场景数据的前提下，实现高效的感知信息融合。
*   **现有方法痛点**：目前的方法通常要求在明文下交换原始数据或详细的检测输出，这会导致严重的隐私泄露（模型提取攻击、数据泄露、跨供应商推断等）。
*   **研究假设**：通过将离散的边界框检测转换为连续的“高斯块（Gaussian Splats）”及其充分统计量，可以利用同态加密的线性特性，在加密域内实现统计意义上的聚合，从而在保护隐私的同时维持感知性能。

---

### 3. 方法设计详解
*   **Pipeline**：
    1.  **供应商端编码**：将检测框转换为高斯分布，并计算加权空间矩向量（即充分统计量，包含位置、不确定性等）。
    2.  **空间分箱（Spatial Binning）**：将空间划分为网格，利用双线性插值将矩向量分配到相邻网格中，并加密传输。
    3.  **服务器端加密融合**：服务器直接对接收到的加密矩向量进行线性求和。
    4.  **接收端重构**：接收方解密并进行逆矩计算，恢复高斯参数，转化为边界框，最后进行基于空间一致性的图合并（Cluster Fusion）得到最终结果。
*   **关键公式**：检测被建模为 $N(\mu_x, \mu_y, \text{diag}(\sigma_x^2, \sigma_y^2))$，其矩向量包含 $\{w, w\mu_x, w\mu_x^2, w\sigma_x^2, \dots \}$ 等统计量，保证了加性同态下的线性可加性。
*   **核心逻辑**：通过预先聚合和分箱，将 $N$ 个检测减少为 $B$ 个空间假设，不仅减少了加密操作次数，还使复杂的多目标关联问题简化为简单的线性求和。

---

### 4. 方法对比分析
*   **本质区别**：传统方案试图在加密域做复杂的非线性匹配（如NMS），而 Sarus 将几何匹配问题转化为统计分布的线性聚合，彻底避开了同态加密不支持复杂非线性运算的限制。
*   **创新贡献**：引入了基于高斯矩的统计表示法，成功实现了感知推理在密文环境下的线性扩展。
*   **适用场景**：适用于存在多方参与、对隐私敏感的自动驾驶协同感知及V2X基础设施系统。

---

### 5. 实验分析
*   **关键结论**：在KITTI数据集上，Sarus 的融合输出与明文融合的 IoU 一致性超过 0.99997，证明了其 numerical correctness。
*   **主要优势**：计算复杂度随网格数量线性扩展，在保持高隐私性的同时，通过互补信息聚合显著提升了长距离和被遮挡场景的检测覆盖率。
*   **主要局限**：HE带来的加密开销依然较大（约 300x-1000x），且目前的实现假设参与方需遵守统一的语义格式。

---

### 6. 实用指南
*   **开源情况**：代码已开源（Github: `mhasan08/sarus`）。
*   **实现细节**：关键参数 $\kappa$（空间扩散参数）和 $\lambda$（包围盒重构系数）对性能影响显著。在数据预处理阶段，务必确保所有参与方使用相同的 `Fcommon` 坐标系。
*   **迁移建议**：该方法可直接迁移至任何基于边界框（Bounding Box）或概率分布输出的感知任务，只需重新定义类别的空间锚点和步长。

---

### 7. 总结
*   **核心思想**：利用高斯分布的充分统计量实现隐私保护下的密文线性聚合。
*   **速记版pipeline**：
    1. 检测转为概率统计分布。
    2. 统计分布加密并传至服务器。
    3. 服务器直接累加密文向量。
    4. 解密并图聚类合并检测。

**Key Findings:**

- In this paper, we present Sarus, a privacy-preserving framework for multi-vendor perception fusion via homomorphic encryption (HE), enabling aggregation without revealing individual vendor outputs.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19146v1)
- [arXiv](https://arxiv.org/abs/2607.19146v1)

---

<a id='2607.19120v1'></a>
## [Latent Riemannian Flow Matching for Geometry-Grounded 3D Foundation Models](https://arxiv.org/abs/2607.19120v1)

**Authors:** Lisa Weijler, Irene Ballester, Guofeng Mei, Tolga Birdal, Pedro Hermosilla

**Published:** 2026-07-21

**Categories:** cs.CV

**Abstract:**

Geometric foundation models, such as the Visual Geometry Grounded Transformer (VGGT), provide strong 3D priors from unposed images. However, such models operate purely in a feed-forward, deterministic regime, \ie~they cannot generate plausible geometry beyond what the input views directly support. Generative models for 3D scenes, on the other hand, must rely on strong geometric priors to produce coherent outputs from sparse inputs. We bridge these two paradigms by performing flow matching directly in VGGT's latent space, leveraging its learned 3D priors without committing to any explicit downstream representation such as Gaussians, meshes, or video-VAE latents. This requires respecting the latent geometry: VGGT tokens occupy a product of high-dimensional hyperspheres on which standard Euclidean flow matching fails. We address this with a Riemannian Flow Matching framework defined on a product manifold of four hyperspheres, aligned with VGGT's multi-scale encoder, which keeps generated tokens on the valid data manifold required by the frozen decoding heads. On RealEstate10K, ScanNet++ and ETH3D, our method achieves strong performance against recent scene generation baselines in both per-view appearance and aggregated 3D geometry, establishing latent-space flow matching on geometric foundation models as a viable paradigm for 3D generation. The project page can be found $\href{https://lisaweijler.github.io/geometry-grounded-rfm/}{\text{here}}$.

**Analysis:**

## 1. 摘要翻译

几何基础模型（如VGGT）能从无姿态图像中提供强大的3D先验，但它们通常在确定性前向传播模式下运行，无法生成输入视图之外的合理几何结构。相比之下，3D场景生成模型必须依赖强大的几何先验才能从稀疏输入中产生连贯的输出。我们通过在VGGT的潜在空间中直接执行流匹配（Flow Matching）来弥合这两个范式，利用其学习到的3D先验，而无需绑定到诸如高斯、网格或视频VAE潜变量等显式下游表示。这要求尊重潜在空间的几何结构：VGGT标记占据了高维超球面的乘积空间，标准的欧几里得流匹配在此空间上会失效。我们通过定义在四个超球面乘积流形上的黎曼流匹配框架解决了这一问题，该框架与VGGT的多尺度编码器对齐，确保生成的标记保持在冻结解码头所需的有效数据流形上。在RealEstate10K、ScanNet++和ETH3D上的实验表明，我们的方法在视角外观和聚合3D几何方面均优于近期场景生成基线，确立了基于几何基础模型的潜在空间流匹配作为3D生成的可行范式。

## 2. 方法动机分析

- **驱动力**：作者旨在利用VGGT这种预训练的几何基础模型，实现既能保留强大几何一致性，又能具备概率生成能力的3D场景重构与新视角合成。
- **痛点**：
    1. 现有生成模型通常缺乏显式的3D先验，需要昂贵的后处理。
    2. VGGT等基础模型是确定性的，无法处理重构的模糊性（即同一稀疏输入可能对应多种合理场景）。
    3. Naive地在VGGT潜在空间应用欧几里得扩散或流匹配会导致“模式崩溃”（Mode Collapse），因为VGGT的token并非分布在欧几里得空间，而是位于特定的超球面流形上。
- **研究假设**：通过在VGGT潜在空间的黎曼流形上进行条件流匹配，可以实现符合几何一致性的高保真3D场景生成。

## 3. 方法设计详解

- **Pipeline**：
    1. **特征提取**：将稀疏无姿态的RGB图像输入冻结的VGGT编码器，获取多尺度潜在token。
    2. **流形建模**：识别VGGT潜在空间为四个零均值超球面的乘积流形 $\mathcal{M} = (\mathcal{S}^{C-2})^4$。
    3. **条件黎曼流匹配（RFM）**：将输入上下文token $c$ 和目标相机姿态 $\pi$ 作为条件，利用条件流匹配学习将噪声映射到目标潜在空间编码 $x_1$ 的路径。
    4. **解码**：生成的潜在编码通过冻结的DPT（Depth）解码头及训练的RGB解码头，重构出深度图、点云和RGB图像。
- **算法解释**：核心在于**黎曼度量下的条件流匹配**。通过在黎曼流形上定义指数映射（Exponential map）和对数映射（Log map），将标准欧几里得向量空间的操作推广到流形，确保训练和推理过程中的路径严格保持在数据的“有效流形”上。

## 4. 方法对比分析

- **本质区别**：不将VGGT视为简单的特征提取器或将其转换到视频潜在空间（如Gen3R），而是直接在VGGT的原生流形上进行生成。
- **创新点**：识别出VGGT潜在空间的超球面几何结构，并将其建模为黎曼流形，实现了流形约束下的生成模型。
- **适用场景**：稀疏视角输入下的场景补全与3D重构，特别适合需要强几何约束的任务。

## 5. 实验分析（精简版）

- **验证方法**：在RE10K、ScanNet++和ETH3D数据集上进行对比，评估指标涵盖深度RMSE、RGB的PSNR/SSIM/FID以及点云Chamfer距离。
- **关键结论**：在ScanNet++和ETH3D上，该方法在几何重构准确性（Chamfer距离降低超过40%）和生成的一致性上均显著优于现有基线。
- **优势**：直接利用预训练模型的先验，减少了计算开销，具备更强的跨场景泛化能力。
- **局限**：目前生成单视角，缺乏多视角联合生成带来的全局一致性（作者列为未来工作）。

## 6. 实用指南

- **实现细节**：
    - **重归一化**：注意训练前将每个块缩放到单位半径，解码前再还原。
    - **流模型训练**：使用 shifted uniform 采样策略偏向噪声端，有助于提升生成质量。
    - **ODE求解**：推理时使用黎曼欧拉法（Riemannian Euler method），20步即可达到平衡。
- **迁移可能**：该框架核心在于流形约束，可迁移至任何基于预训练特征（如DINOv2）且具有明确流形结构的生成任务中。

## 7. 总结

- **核心思想**：通过黎曼流形流匹配，在冻结的几何模型潜在空间上直接生成一致性3D场景。
- **速记版pipeline**：
    1. 用VGGT编码稀疏视角图像。
    2. 将token投影至超球面乘积流形。
    3. 训练条件黎曼流模型预测路径。
    4. 用冻结解码器转译为RGB/深度图。

**Key Findings:**

- On RealEstate10K, ScanNet++ and ETH3D, our method achieves strong performance against recent scene generation baselines in both per-view appearance and aggregated 3D geometry, establishing latent-space flow matching on geometric foundation models as a viable paradigm for 3D generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19120v1)
- [arXiv](https://arxiv.org/abs/2607.19120v1)

---

<a id='2607.19036v1'></a>
## [CoGoal3D: Collaborative 3D Object Detection with 3D-Aware Fusion and Refinement](https://arxiv.org/abs/2607.19036v1)

**Authors:** Zhihao Yang, Zhiyu Xiang, Peng Xu, Tianyu Pu, Kai Wang, Eryun Liu, Dongping Zhang, Yong Ding

**Published:** 2026-07-21

**Categories:** cs.CV, cs.AI

**Abstract:**

V2X collaborative object detection features overcoming the limitations of single-vehicle systems by aggregating environmental features from multiple collaborative agents. However, existing mainstream V2X perception methods mainly focus on 2D BEV object detection. When 3D detection task is concerned, inferior results are obtained because they ignore the 3D spatial misalignment caused by differing height and attitude among the collaborators. In this paper, we propose a novel collaborative 3D object detection framework called CoGoal3D, which extracts and refines the 3D feature gradually in a two-stage pipeline. In the first stage, a multiscale 3D-aware global fusion module is designed to mitigate the 3D spatial misalignment. The resulting proposals are then refined in the second stage with an auxiliary task of 3D point reconstruction. An effective multi-agent collaborative data augmentation strategy is further proposed to enrich the training data while minimizing information loss. Extensive experiments on public real-world datasets demonstrate that our CoGoal3D achieves new state-of-the-art performance, with 3D AP@0.7 improvements of 10.86%, 10.34%, and 10.18% on the DAIR-V2X, V2V4Real, and V2X-Real datasets, respectively. Code is available at https://github.com/Megalo-f/CoGoal3D.

**Analysis:**

### 1. 摘要翻译
V2X协作式目标检测通过聚合多个协作智能体的环境特征，克服了单车系统的局限性。然而，现有主流V2X感知方法主要聚焦于2D BEV目标检测。在3D检测任务中，由于忽略了协作智能体之间高度和姿态差异导致的3D空间错位，这些方法往往只能获得较差的结果。本文提出了一种新颖的协作式3D目标检测框架CoGoal3D，通过两阶段流水线逐步提取并细化3D特征。在第一阶段，设计了多尺度3D感知全局融合模块以缓解3D空间错位；第二阶段通过3D点重构的辅助任务对初始方案进行细化。此外，提出了一种高效的多智能体协作数据增强策略，在最小化信息损失的同时丰富了训练数据。在公开真实世界数据集上的广泛实验表明，CoGoal3D达到了新的SOTA性能，在DAIR-V2X、V2V4Real和V2X-Real数据集上，3D AP@0.7分别提升了10.86%、10.34%和10.18%。代码已开源。

### 2. 方法动机分析
- **核心驱动力**：解决V2X感知中因不同智能体间传感器高度、俯仰角等姿态差异导致的3D空间严重错位问题。
- **痛点**：现有主流的基于广播（Broadcast）的融合方法仅依赖2D BEV warping，假设所有智能体处于同一水平面，这在3D空间中是失效的。
- **核心直觉**：3D感知需要显式的3D空间对齐和精细的几何信息建模，而不仅仅是2D平面的特征拼接。

### 3. 方法设计详解
CoGoal3D采用两阶段Pipeline：
- **Stage 1 (多尺度3D感知全局融合 3D-AGF)**：
  - **3D Position Encoding**：将不同智能体的特征投影到统一坐标系，利用3D pillar中心坐标及其与参考点的相对位置，通过MLP显式注入3D位置信息。
  - **3D-Aware Deformable Cross Attention**：不同于传统的BEV Warp，利用可变形注意力（Deformable Attention）动态学习采样点，补偿因姿态差异导致的采样位置偏移，实现跨智能体的特征对齐。
- **Stage 2 (重构引导的局部细化 RGLR)**：
  - **RoI-level 3D Point Reconstruction**：在BEV proposals的基础上，使用辅助头进行3D点云重构。
  - **Ground Truth Optimization (GTO)**：这是该模块的关键。由于原始raw点云存在标定误差或同步问题，通过匹配GT框并计算相对变换，将协作端的点云“对齐”到ego坐标系，为重构任务提供高质量的监督信号。

### 4. 方法对比分析
- **本质区别**：从传统的“2D BEV特征融合”转向“显式3D位置感知+几何信息重构监督”的融合范式。
- **核心创新**：
  1. **3D-AGF模块**：通过可变形注意力实现动态对齐，打破了平坦地面的假设。
  2. **GTO机制**：解决了协作端点云在GT层面的空间不一致性，提升了监督信号的质量。
  3. **MCDA增强**：通过局部旋转+全局增强组合，规避了传统方法（如DPTP）在增强过程中导致的视场（FOV）信息丢失问题。

### 5. 实验分析
- **验证方法**：在DAIR-V2X、V2V4Real、V2X-Real数据集上与SOTA方法（如DI-V2X, CoBEVT等）进行对比。
- **关键结论**：在DAIR-V2X上3D AP@0.7提升显著（>10%），说明针对3D几何对齐的改进对长距离、强空间相关性的3D任务至关重要。
- **优势**：鲁棒性高，在处理定位误差和高延迟情况下依然稳健。
- **局限**：模型引入两阶段结构和重构辅助任务，使得训练过程比单阶段模型更复杂。

### 6. 实用指南
- **开源地址**：`https://github.com/Megalo-f/CoGoal3D`
- **实现细节**：GTO方法依赖于匈牙利匹配算法，训练时需注意对齐GT。MCDA增强策略建议优先尝试，其通过局部旋转规避信息丢失的思路可迁移至大多数协作感知框架。
- **迁移建议**：3D-AGF模块作为一种增强特征融合的通用插件，可尝试集成到基于Transformer的感知框架中。

### 7. 总结
- **核心思想**：通过显式3D位置建模与几何重构引导，实现跨视角特征的精确对齐。
- **速记版pipeline**：
  1. 多智能体特征编码；
  2. 基于3D位置编码与可变形注意力融合特征；
  3. 初步生成Proposal；
  4. 利用GTO修正GT并执行辅助点重构进行细化。

**Key Findings:**

- In this paper, we propose a novel collaborative 3D object detection framework called CoGoal3D, which extracts and refines the 3D feature gradually in a two-stage pipeline.
- Extensive experiments on public real-world datasets demonstrate that our CoGoal3D achieves new state-of-the-art performance, with 3D AP@0.7 improvements of 10.86%, 10.34%, and 10.18% on the DAIR-V2X, V2V4Real, and V2X-Real datasets, respectively.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19036v1)
- [arXiv](https://arxiv.org/abs/2607.19036v1)

---

<a id='2607.18840v1'></a>
## [WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory](https://arxiv.org/abs/2607.18840v1)

**Authors:** Haisheng Su, Zongdai Liu, Xin Jin, Haoxuan Dou, Chengming Hu, Baorun Li, Zhanwang Liu, Ruiyan Xu, Jianjie Fang, Xin Zhang, Zhenjie Yang, Xue Yang, Chen Gao, Junchi Yan, Yong Li, Wei Wu

**Published:** 2026-07-21

**Categories:** cs.RO

**Abstract:**

World Action Models (WAMs) offer a promising paradigm for robotic manipulation by jointly modeling visual state transitions and robot actions. However, existing WAMs are constrained by limited temporal context, coarse episode-level language supervision, and predominantly text-only conditioning, which hinder task-progress tracking and fine-grained language-video-action grounding while limiting visual-context reasoning and cross-embodiment transfer. In this paper, we introduce WorldScape Policy 2.0, a controllable WAM with reasoning-augmented long short-term memory. Its causal short-term visual memory supplies recent observations as DiT prefill to preserve local interaction dynamics, while its long short-term event memory organizes historical VLM outputs into global-history, local-active, and event-boundary representations for progress-aware retrieval. The retrieved history augments perception and autoregressively generated planning tokens, yielding an implicit subgoal condition for autonomous planning; semantic forcing further transfers event-level instruction semantics into this latent planning pathway. To establish fine-grained multimodal controllability, we construct ManipEvent-5M, an event-grounded embodied pretraining dataset containing nearly 5 million event segments with aligned action trajectories, episode-level task instructions, segment-level subtask captions, goal images, and video demonstrations. These designs provide a unified interface for autonomous planning from high-level instructions and controllable execution from fine-grained text, goal-image, or video-context prompts. Experiments in both simulation and real-world platforms demonstrate superior capabilities in long-horizon autonomous planning, fine-grained instruction following and in-context adaptation.

**Analysis:**

### 1. 摘要翻译
世界动作模型（WAMs）通过联合建模视觉状态转移和机器人动作，为机器人操作提供了有前景的范式。然而，现有的WAM受限于有限的时间上下文、粗糙的片段级语言监督以及以文本为主的条件设定，这阻碍了任务进度跟踪和细粒度语言-视频-动作对齐，并限制了视觉上下文推理和跨形态迁移。本文介绍了 **WorldScape Policy 2.0**，一个具有推理增强长短期记忆的可控WAM。其因果短期视觉记忆提供近期观测作为DiT预填充以保持局部交互动态，而其长期短时事件记忆将历史VLM输出组织为全局历史、局部活跃和事件边界表示，用于进度感知检索。检索到的历史增强了感知和自回归生成的规划标记，产生了用于自主规划的隐含子目标条件；语义强制将事件级指令语义进一步转移到此潜在规划路径中。为建立细粒度多模态可控性，我们构建了 **ManipEvent-5M**，一个事件驱动的具身预训练数据集，包含近500万个带有对齐动作轨迹、片段级子任务标注、目标图像和视频演示的事件片段。这些设计为从高级指令进行自主规划和从细粒度文本、目标图像或视频上下文提示进行可控执行提供了统一接口。在模拟和真实世界平台上的实验表明，该模型在长程自主规划、细粒度指令遵循和上下文内适应方面具有卓越能力。

---

### 2. 方法动机分析
*   **驱动力**：解决现有WAM在长程操作中因缺乏历史感知和细粒度控制而导致的任务执行能力弱的问题。
*   **痛点**：当前模型多仅基于当前帧或极短窗口，无法理解任务进度（例如，难以区分同一个桌面上不同阶段的动作）；此外，缺乏对非语言形式（目标图像、视频演示）的直接利用。
*   **研究假设**：长程可控性需要两类互补的上下文：语义事件记忆用于推理任务进度，帧级视觉记忆用于保持局部交互动态。

---

### 3. 方法设计详解
*   **pipeline**：
    1.  **事件记忆分支（VLM-based）**：将历史Chunks压缩为“全局历史”、“局部活跃”和“事件边界”三个维度，并基于当前任务需求进行检索。
    2.  **视觉记忆分支（DiT-based）**：将近期观测序列作为因果视觉预填充（Prefill），直接接入DiT保持局部动态。
    3.  **语义强制（Semantic Forcing）**：利用训练阶段细粒度Caption的T5 Embedding作为监督信号，通过“语义强制”将事件层面的语义注入到自主规划的潜在空间中。
    4.  **统一预测**：将记忆信息、提示词（文本/图像/视频）和视觉状态融合，输入Causal DiT同时输出未来的视频潜在表示和动作轨迹。
*   **算法核心**：利用语义改变检测公式，通过余弦相似度计算前后Chunk的差异，筛选关键转折点（Event-Boundary），从而在保持稀疏性的前提下保留关键信息。

---

### 4. 方法对比分析
*   **本质区别**：与仅基于视频预测的WAM不同，WorldScape Policy 2.0引入了**显式的语义事件记忆**与**隐式潜在子目标规划**的双重机制。
*   **创新点**：
    *   **Reasoning-Augmented Memory**：首次在WAM中系统化地解耦了“语义进展感知”与“动作动态保持”。
    *   **ManipEvent-5M**：提供了500万级别的数据基座，支持复杂的跨模态指令。
*   **适用场景**：适用于需要长程决策、多阶段拆解以及跨形态演示学习（Human-to-Robot）的具身智能任务。

---

### 5. 实验分析
*   **关键结论**：在RoboTwin 2.0基准测试中，平均成功率达到94.3%，显著优于同类VLA和WAM baseline。
*   **核心优势**：在Out-of-Domain（OOD）任务中表现出更强的泛化性（60% OOD成功率），证明了其多模态对齐和事件级预训练的有效性。
*   **局限**：模型依赖于大规模视频数据的预处理，对于实时性要求极高的场景，VLM推理分支可能引入额外的延迟。

---

### 6. 实用指南
*   **开源情况**：提供项目主页（GitHub地址已在原文给出）。
*   **实现建议**：
    *   **超参数**：语义强制的loss系数（λs）需谨慎调整，文中为0.001。
    *   **训练细节**：采用了三阶段训练策略（事件预训练 -> 记忆感知中继 -> 下游微调），这一流程对模型性能至关重要。
*   **迁移迁移**：其DiT架构和适配器（Adapter）设计允许通过添加少量的特定形态适配器，快速迁移到不同的机器人硬件平台上。

---

### 7. 总结
*   **核心思想**：通过双模态记忆（语义事件+视觉细节）实现任务进展感知与精准动作控制。
*   **速记pipeline**：
    1.  **数据清洗**：利用Qwen-VL将长视频拆解为语义对齐的事件片段。
    2.  **记忆构建**：通过Attention Pooling压缩历史信息并提取关键转折点。
    3.  **双流注入**：视觉流做短期预测，语义流做长程规划。
    4.  **端到端协同**：在DiT中融合所有模态，联合输出动作和视觉轨迹。

**Key Findings:**

- In this paper, we introduce WorldScape Policy 2.0, a controllable WAM with reasoning-augmented long short-term memory.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.18840v1)
- [arXiv](https://arxiv.org/abs/2607.18840v1)

---

