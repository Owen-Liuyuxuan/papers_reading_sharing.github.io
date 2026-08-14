time: 20260814

# Arxiv Computer Vision Papers - 2026-08-14

## Executive Summary

## 每日计算机视觉论文执行摘要（2026-08-13）

本日10篇论文整体呈现出三大特点：**基准测试密集涌现**、**具身智能与世界模型持续升温**、**视频/多模态生成与理解走向长期化与细粒度化**。

### 1. 主要主题与趋势

- **全新基准与评测体系成为主线**：多篇论文聚焦构建更贴近真实应用场景的基准，如人体运动跟踪（HumanTracker）、长期世界模型目标（PlayWorld）、科学图表编辑（Edit2TikZ）、月级自我中心视频记忆（EgoMonth）以及物理接地的人形机器人导航（HumanoidVLN）。这反映出社区正从“模型刷榜”转向“更全面、更人类对齐、更长期”的评测。
- **具身智能与自动驾驶并进**：自动驾驶领域出现几何约束的统一3D感知；机器人领域则出现基于物理仿真的人形机器人视觉-语言导航，以及利用RGB-D视频生成改善人-机器人物体交接预测。视觉不再只是“看”，而是为“行动”服务。
- **视频生成与潜空间建模走向深入**：有论文重新审视视频潜空间设计（V-RAE），也有工作探索基于提示的红外-可见光图像融合（P2Fusion），显示底层视觉表示学习仍是活跃方向。

### 2. 重要或创新性论文

- **《Edit2TikZ》**（Zhang等）——首个针对科学图表编辑的TikZ格式综合基准，将图表编辑与代码生成、结构理解结合，开辟了“文档级视觉编辑”新方向，具有较强新颖性。
- **《HumanoidVLN》**（Pham等）——提出物理接地的仿真器与基准，支持多样人形具身进行视觉-语言导航，直接连接VLN与具身物理现实，是迈向通用机器人智能的重要一步。
- **《V-RAE》**（Guo, Wu, Fei）——重新思考视频生成中的潜空间设计，题名直指核心架构问题，可能影响下一代视频生成模型的基础设计。
- **《EgoMonth》**（Chen等）——将自我中心视频记忆从“小时/天”扩展到“月”级，对长期时空记忆建模和持续学习具有独特价值。

### 3. 新兴研究方向与技术

- **“行为/动作作为视觉约束或监督”**：《Attention from Action》提出从动作中产生视觉瓶颈，代表一种具身感知与策略学习深度耦合的趋势。
- **物理接地与多模态具身仿真**：HumanoidVLN强调“physics-grounded”，说明仿真真实度成为提升视觉导航、操作等能力的关键瓶颈。
- **长期视频理解与生成统一**：从EgoMonth到PlayWorld，都在挑战模型对长时间跨度、长期目标的理解与预测，可能是未来“世界模型”落地的核心能力。
- **图结构/程序表示的视觉任务融合**：Edit2TikZ使用TikZ（一种程序化绘图语言）作为中间表示，提示“视觉-代码”联合建模正成为处理结构化视觉内容的新范式。

### 4. 最值得精读的论文

若时间有限，建议优先阅读：

1. **HumanoidVLN**——具身智能与物理仿真结合的代表，对做机器人、视觉导航的研究者价值最高。
2. **Edit2TikZ**——新颖任务与基准，对多模态大模型、图表理解/生成方向有很强启发性。
3. **V-RAE**——视频生成基础表示的关键反思，适合关注扩散模型/潜空间优化的读者。
4. **EgoMonth**——长期视频理解的前沿基准，适合做视频记忆、持续学习相关工作。
5. **Attention from Action**——篇幅可能较短但思想性强，适合对具身感知与策略学习交叉感兴趣的读者。

---

## Table of Contents

1. [HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark](#2608.13555v1)
2. [PlayWorld: Benchmarking World Models with Agent Players over Long-Horizon Objectives](#2608.13552v1)
3. [Edit2TikZ: A Comprehensive and Challenging Benchmark for Scientific Figure Editing with TikZ](#2608.13441v1)
4. [Attention from Action, for Action: Emergent Visual Bottlenecks for Policy Learning](#2608.13422v1)
5. [Geometry-Grounded Unified 3D Perception for Autonomous Driving](#2608.13147v1)
6. [EgoMonth: A Month-Level Egocentric Video Benchmark for Long-Term Spatiotemporal Memory](#2608.13113v1)
7. [P2Fusion: Prompt-based Progressive Infrared-Visible Image Fusion via Dual-Prior Distillation](#2608.13045v1)
8. [RGB-D Video Generation for Improving Human-to-Robot Object Handover Prediction](#2608.13028v1)
9. [HumanoidVLN: A Physics-Grounded Simulator and Benchmark for Vision-Language Navigation Across Diverse Humanoid Embodiments](#2608.12860v1)
10. [V-RAE: Rethinking Video Latent Spaces for Generation](#2608.13556v1)

---

## Papers

<a id='2608.13555v1'></a>
## [HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark](https://arxiv.org/abs/2608.13555v1)

**Authors:** Dairu Liu, Zekun Qi, Jiayu Zeng, Ruixi Yu, Yu Guan, Yintianrun Zhang, Xuchuan Chen, Sikai Liang, Zekai Li, Chenghuai Lin, Xinqiang Yu, Wenyao Zhang, He Wang, Li Yi

**Published:** 2026-08-13

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Humanoid motion tracking is central to teleoperation and whole-body imitation, yet evaluation often disagrees with what people perceive in videos. Kinematic errors average per-frame pose differences but miss the physical artifacts that matter most, particularly unstable support and incorrect contacts such as foot skating and mistimed touch-downs. Meanwhile, widely used test suites are small and lack the diversity needed to stress contact-rich, long-horizon behaviors. We introduce HumanTracker to make humanoid tracking evaluation both perceptually aligned and scalable. The HumanTracker benchmark contains approximately 153 hours of optical motion trajectories from multiple professional performers, organized into four motion families with text labels for fine-grained diagnosis. We further propose HumanScore, a preference-aligned metric trained on 12K motion pairs containing 24K motions. Across representative state-of-the-art trackers, HumanScore better predicts human preferences and reveals contact and stability failures that kinematic metrics often miss.

**Analysis:**

以下是对《HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark》论文的深入分析：

### 1. 摘要翻译
类人机器人运动跟踪对远程操控和全身模仿至关重要，但现有评估指标与人类视觉感知往往不一致。运动学指标平均每帧的位姿差异，却忽略了至关重要的物理伪影，特别是支撑不稳定、足部打滑及触地时机错误等问题。此外，现有的测试集规模较小，缺乏处理接触丰富、长跨度行为所需的各种场景。我们引入了HumanTracker，旨在使类人运动跟踪评估既符合人类感知，又具备可扩展性。该基准包含约153小时的专业演员动作轨迹，按四个动作族分类并配有文本标签，以便进行精细诊断。此外，我们提出了HumanScore，这是一种在12K运动对（共24K条动作）上训练的偏好对齐指标。在现有的代表性最先进追踪器中，HumanScore能更好地预测人类偏好，并揭示了运动学指标常忽略的接触和稳定性失败。

### 2. 方法动机分析
*   **驱动力**：解决现有“运动跟踪评估与真实人类感知脱节”的问题。
*   **现有痛点**：传统方法如MPJPE（平均关节点位置误差）将跟踪视为帧级位姿匹配，忽略了**时间连贯性、接触物理约束及稳定性**。两段MPJPE相似的动作，在人类观察者眼中可能一段稳定，一段却频繁出现足部打滑或抖动。
*   **研究假设**：通过在人类偏好数据上训练一个能捕捉时序依赖特征的奖励模型（Reward Model），可以比纯数值指标更准确地评估跟踪质量。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据采集与处理**：构建包含153小时动作数据的HumanTracker基准，涵盖日常、动态、交互、地面四类动作，通过通用运动重定向（GMR）标准化到机器人形态。
    2.  **偏好数据构建**：对不同追踪器输出的同一参考动作的Rollout（轨迹）进行两两配对，邀请专家进行人类偏好标注，形成训练集。
    3.  **HumanScore模型设计**：
        *   **输入编码**：将5s的窗口（539维/帧，含参考状态、机器人状态、触点动态、根节点运动等）输入模型。
        *   **Transformer处理**：利用Transformer捕捉时序上的物理伪影，配合掩码（Mask）处理变长序列。
        *   **Reward Head**：通过Bradley-Terry loss优化，预测一个标量值作为HumanScore。
*   **模型结构**：使用双向Transformer架构，采用Pooling层将帧序列聚合为 trajectory representation，随后经MLP映射为标量奖励。
*   **关键公式**：$\mathcal{L} = -\log \sigma(r_{chosen} - r_{rejected})$。该损失函数强迫模型学习如何对人类偏好的轨迹给予更高奖励。

### 4. 方法对比分析
*   **本质区别**：从“基于规则的计算”转向“基于人类感知的学习评估”，从“静态帧评估”转向“动态时序序列评估”。
*   **创新贡献**：引入了大规模且精细分类的HumanTracker数据集；提出了基于偏好的HumanScore，能够敏锐捕捉物理不稳定性（如足部打滑、抖动）。
*   **适用场景**：适用于机器人运动跟踪策略的对比评估、算法迭代中的性能基准测试。

### 5. 实验分析
*   **验证方法**：在四个动作类上对GMT、TWIST2、SONIC、Humanoid-GPT进行测试，并与传统指标进行对齐率（Align Rate）对比。
*   **关键结果**：HumanScore与人类偏好的一致性显著高于传统指标（如MPJPE、关节加速度/抖动），在揭示接触类失败方面表现尤为突出。
*   **优势**：捕捉了复杂的长跨度物理 artifacts；能够量化评估稳定性。
*   **局限**：作为奖励模型，若直接用于强化学习优化可能导致模型“作弊”（利用奖励函数的缺陷产生虚假高质量动作），需额外正则化。

### 6. 实用指南
*   **开源情况**：已开源代码库与项目主页（参考OCR页）。
*   **实现细节**：数据窗口设为250帧（5s），使用AdamW优化器，训练20轮；输入向量必须包含精确的接触动态特征。
*   **迁移可能**：HumanScore的设计理念可直接迁移至其他机器人任务评估，如操纵臂抓取、复杂场景导航等。

### 7. 总结
*   **核心思想**：利用人类偏好训练时序Transformer，让机器人评估更“懂”人类。
*   **速记版Pipeline**：
    1. 收集海量动作及机器人Rollout数据。
    2. 专家对轨迹对进行偏好标注。
    3. 训练Transformer预测偏好得分。
    4. 使用该得分自动评估新策略轨迹。

**Key Findings:**

- We introduce HumanTracker to make humanoid tracking evaluation both perceptually aligned and scalable.
- Across representative state-of-the-art trackers, HumanScore better predicts human preferences and reveals contact and stability failures that kinematic metrics often miss.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.13555v1)
- [arXiv](https://arxiv.org/abs/2608.13555v1)

---

<a id='2608.13552v1'></a>
## [PlayWorld: Benchmarking World Models with Agent Players over Long-Horizon Objectives](https://arxiv.org/abs/2608.13552v1)

**Authors:** Kaixin Ding, Xi Chen, Minghong Cai, Zhiyuan Xu, Yiyang Wang, Yuxiang Lu, Junyi Li, Shuyang Chen, Yuan Gao, Xin Tao, Pengfei Wan, Hengshuang Zhao

**Published:** 2026-08-13

**Categories:** cs.CV

**Abstract:**

Video world models simulate future states conditioned on current observations and user actions. Recent systems have demonstrated impressive video consistency and action controllability over long sequences. However, fairly comparing these interactive models remains challenging. In practice, a human player typically evaluates a world model by pursuing long-horizon objectives through interaction. For example, a user may turn around 360 degrees to see whether the environment remains consistent, or walk into the water and inspect whether realistic water ripples are generated. The action sequence required to achieve the same objective may vary substantially between models, making fixed action-conditioned evaluation unsuitable for cross-model comparison. To address this, we employ multi-modal Agent Players to interact with world models toward specified long-horizon objectives. Building on this paradigm, we introduce PlayWorld, a benchmark providing 171 scenarios, each with a specified objective. To evaluate performance thoroughly, we assess models along four core dimensions: geometry consistency, interaction fidelity, out-of-sight evolution, and insight evolution. In addition, we incorporate basic ability metrics for video quality and controllability. Experiments across nine state-of-the-art world models reveal that current models remain unreliable on long-horizon interactive objectives, particularly in maintaining spatial consistency and persistent state evolution. Code and data are available at https://github.com/kxding/PlayWorld.

**Analysis:**

### 1. 摘要翻译
视频世界模型根据当前观察和用户动作模拟未来状态。尽管现有系统在长序列视频一致性和动作可控性方面表现出色，但公平比较这些交互式模型仍然极具挑战。在实践中，人类玩家通常通过交互追求长程目标来评估模型。例如，用户可能会旋转360度以检查环境一致性，或走进水中观察是否生成真实的涟漪。实现同一目标所需的动作序列可能因模型而异，使得基于固定动作的评估不适用于跨模型比较。为解决此问题，我们引入了PlayWorld，一个包含171个指定目标场景的基准测试。为了全面评估性能，我们从几何一致性、交互保真度、视外演化和洞察演化四个核心维度评估模型。此外，我们整合了视频质量和可控性的基础能力指标。对九个最先进世界模型的实验表明，当前模型在长程交互目标上仍然不可靠，特别是在保持空间一致性和持久状态演化方面。

### 2. 方法动机分析
- **驱动力**：旨在填补交互式视频世界模型评估的空白，建立符合人类真实评估方式的自动化框架。
- **痛点**：现有基准（如VBench、WorldScore）多依赖“预定义轨迹”，由于不同模型动作粒度和响应速度不同，固定指令会导致相同控制无法达到相同预期状态，评估指标存在偏差。
- **研究假设**：通过引入具有“闭环自适应”能力的AI智能体（Agent Player），可以在评估过程中动态调整动作序列，从而在不同模型间维持一致的评估意图。

### 3. 方法设计详解
- **核心组件**：
    1.  **Agent Player**：结合预设的“基础动作序列”作为参考，通过观察每一帧生成的画面，动态做出“Keep（保持）、Stop（提前停止）、Extend（延长时间）、Correct（修正动作）、End（终止）”的决策。
    2.  **VQA Rubric Verifier**：利用Gemini 3.1 Pro作为评分器。在评分前进行“维度特定验证（如轨迹有效性、主体可达性）”，确保 rollout 真正执行了目标动作，随后根据结构化 rubric 给出1-5分的量化指标。
- **工作流程**：
    1.  输入初始图像、长程目标及参考动作序列。
    2.  Agent接口将动作转化为模型原生控制，生成视频片段。
    3.  Agent模型根据最新帧和历史状态调整下一步操作（闭环交互）。
    4.  达到终止条件后，利用VQA verifier对交互结果进行多维度（几何、交互、视外、洞察）评分。

### 4. 方法对比分析
- **本质区别**：从“静态轨迹执行”转变为“动态闭环交互”。
- **创新贡献**：提出了一套能够适应异构模型动作粒度差异的动态评估协议，且通过VQA verifier实现了对复杂物理交互（如碰撞、物体留存）的语义级评估。
- **适用场景**：适用于评估任何支持交互的视频生成模型或世界模型。

### 5. 实验分析（精简版）
- **验证方法**：在171个标注场景上测试了9个主流模型，并与人工评估结果进行相关性校验（Spearman相关性高）。
- **关键结论**：当前的顶尖模型在处理简单的视频质量任务时表现良好，但在“长程空间一致性”和“持续状态演化（如物体消失后再出现或动作的持续演进）”上表现显著不足。
- **优势/局限**：优势在于评价体系的高度拟人化与一致性；局限在于高度依赖高性能多模态大模型作为评分器（VQA），且评估成本较高。

### 6. 实用指南
- **开源情况**：代码和数据已开源，详见 [PlayWorld官网](https://kxding.github.io/project/PlayWorld/)。
- **迁移可能**：该框架的Agent-based评估范式可直接迁移至机器人模拟器、自动驾驶仿真等任务中。
- **注意细节**：在本地执行 chunk-wise 生成时，需确保模型支持非空动作输入，以适配对passive observation的需求。

### 7. 总结
- **核心思想**：通过智能体闭环交互，实现跨模型的长程目标公平评估。
- **速记版pipeline**：
    1. 指定长程目标与基础参考动作；
    2. Agent模型观察输出并实时动态调整指令；
    3. 执行闭环交互，直至达到目标或预算耗尽；
    4. 通过多维度VQA校验评分，量化评估模型能力。

**Key Findings:**

- Building on this paradigm, we introduce PlayWorld, a benchmark providing 171 scenarios, each with a specified objective.
- Experiments across nine state-of-the-art world models reveal that current models remain unreliable on long-horizon interactive objectives, particularly in maintaining spatial consistency and persistent state evolution.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.13552v1)
- [arXiv](https://arxiv.org/abs/2608.13552v1)

---

<a id='2608.13441v1'></a>
## [Edit2TikZ: A Comprehensive and Challenging Benchmark for Scientific Figure Editing with TikZ](https://arxiv.org/abs/2608.13441v1)

**Authors:** Zongyun Zhang, Jiacheng Ruan, Xian Gao, Ruizhu Zhou, Lingcheng Meng, Lining Hu, Ting Liu, Yuzhuo Fu

**Published:** 2026-08-13

**Categories:** cs.CV

**Abstract:**

Although multimodal large language models (MLLMs) have shown substantial potential in visual understanding and graphic code generation, editing scientific figures through code presents a greater challenge: a model must jointly recover visual structure, ground the requested change, generate compilable code, and preserve all unrelated content. While existing TikZ benchmarks mainly focus on figure reconstruction and generation, few systematically evaluate instruction-guided scientific figure editing with compilable code. We introduce Edit2TikZ, a comprehensive benchmark for scientific figure editing tasks, featuring 1,548 diverse and high-quality samples. Edit2TikZ combines real-world and controlled synthetic edit cases, supports both textual and visual localization request, and contains multi-step editing, each with step-level annotations. We further construct a human-aligned evaluation framework to measure whether a requested edit is completed while irrelevant content is preserved. Utilizing Edit2TikZ, we evaluate 14 mainstream MLLMs and find that current systems remain unreliable: on average, proprietary models achieve a compilation success rate of merely 75% and remain limited in both figure restoration and edit correctness, while compact models below 9B struggle further with instruction following and complete figure generation. Therefore, we build a mixed training set TikZEditMix and adopt reconstruction-then-editing curriculum learning for compact models. On Qwen3.5-4B, this training improves the compilation success rate from 45.35% to 83.40% and yields an average improvement of 18.7 points across our proposed evaluation metrics. The code and data will be released at https://github.com/Solunny/Edit2TikZ.

**Analysis:**

### 1. 摘要翻译
尽管多模态大语言模型（MLLMs）在视觉理解和代码生成方面展现了潜力，但在基于代码的科学图表编辑方面仍面临巨大挑战：模型必须联合恢复视觉结构、锚定编辑请求、生成可编译代码，并保持非目标内容的完整性。针对现有TikZ基准测试仅侧重于重构而非编辑的局限，本文提出了**Edit2TikZ**，这是一个包含1,548个样本的科学图表编辑基准。我们构建了一个人类对齐的评估框架，以衡量编辑准确性与非目标内容保持度。评估结果表明，当前主流模型（包括最强闭源模型）在编辑任务中表现欠佳。为此，我们提出了**TikZEditMix**数据集及重构-编辑两阶段课程学习策略。实验证明，该方法使Qwen3.5-4B模型的编译成功率从45.35%提升至83.40%，核心指标提升18.7点。

### 2. 方法动机分析
*   **驱动力**：实现端到端的科学图表智能编辑，即输入原始图像和编辑指令，输出可编译的TikZ代码。
*   **痛点**：现有工作大多关注“图表生成”或“重构”，缺乏对“在维持原有结构前提下进行局部修改”的系统性评估与训练。
*   **核心直觉**：编辑任务不仅是代码生成，更是对视觉结构的深度重构与理解；单纯的code-level相似度评估（如cBLEU）不能反映编辑的实际功能正确性。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **数据采集与处理**：从arXiv源文件中提取TikZ代码，通过渲染、归一化及人工筛选构建三类样本：真实编辑、文本合成编辑、视觉定位编辑。
    2.  **两阶段课程学习**：
        *   **阶段一（重构）**：训练模型由图到代码的直接映射（即恢复完整TikZ结构）。
        *   **阶段二（编辑）**：引入指令及编辑操作（编辑单元$E$），训练模型执行指令并保证其余结构不变。
    3.  **多级评估体系**：设计了**RestorationScore (RS)**（非目标区域保持度）和**EditCorrectnessScore (ECS)**（指令完成度），通过人类对齐的指标替代不可靠的图像相似度（DSim）。
*   **算法解释**：引入了视觉定位提示 ($q_{vis}$)，利用覆盖在源图上的半透明红框显式引导模型关注编辑区域，减少模型定位歧义。

### 4. 方法对比分析
*   **本质区别**：从传统的单纯“生成式”转向“编辑式+保真式”双重训练目标。
*   **创新贡献**：提出首个支持多操作类型、带视觉定位、且包含人类对齐评估指标的科学图表编辑基准。
*   **适用场景**：适用于需要精确控制科学图表细节（如电路图、几何图形、模型架构）的AI系统。

### 5. 实验分析
*   **关键结果**：现有模型在复杂编辑任务上表现出显著的“灾难性遗忘”或“结构破坏”。两阶段课程学习策略能显著提升Compact模型的鲁棒性（4B模型ECS提升14.78点）。
*   **主要局限**：模型在处理dense relational structure（如复杂的依赖关系）时极易出现断层或局部错误；模型参数规模对长期结构的维护仍有影响。

### 6. 实用指南
*   **开源与实现**：代码与数据集已公开（https://github.com/Solunny/Edit2TikZ）。
*   **训练建议**：
    *   **两阶段初始化**：先进行广泛的重构训练，再进行特定的编辑任务微调，是提升小参数模型性能的关键。
    *   **反馈回路**：在推理时采用Agentic迭代修正（利用LaTeX编译报错信息进行多次尝试）能显著改善结果。
*   **迁移建议**：该框架的“重构+特定任务编辑”范式可迁移至其他结构化图形编程任务（如SVG或Matplotlib编辑）。

### 7. 总结
*   **核心思想**：通过分阶段课程学习，将重构与局部精准编辑能力解耦，实现科学图表的高质量智能编辑。
*   **速记版Pipeline**：
    1.  构建多样化科学编辑数据集。
    2.  阶段一训练：先从图像直接恢复源码。
    3.  阶段二训练：注入指令以执行局部编辑。
    4.  评估与对齐：使用RS与ECS衡量编辑效果，而非图像像素相似度。

**Key Findings:**

- We introduce Edit2TikZ, a comprehensive benchmark for scientific figure editing tasks, featuring 1,548 diverse and high-quality samples.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.13441v1)
- [arXiv](https://arxiv.org/abs/2608.13441v1)

---

<a id='2608.13422v1'></a>
## [Attention from Action, for Action: Emergent Visual Bottlenecks for Policy Learning](https://arxiv.org/abs/2608.13422v1)

**Authors:** Zheyu Zhuang, Ruiyu Wang, Nick Heppert, Johannes Fabian Hahn, Abhinav Valada, Florian T. Pokorny, Danica Kragic

**Published:** 2026-08-13

**Categories:** cs.RO

**Abstract:**

Visual bottlenecks that focus policy inputs on regions of interest (ROIs) can improve data-efficient visuomotor learning by separating where to look from how to act. Many ROI interfaces rely on external spatial labels, such as gaze, object classes, or affordance annotations. Label-free alternatives often derive crops from trajectories by detecting gripper or motion events and centering a fixed crop at the projected end-effector. Such action-derived crops are useful spatial priors that require no additional labels, but they encode fixed choices about event timing, proxy points, and crop scale. When the visual evidence needed for control lies away from the end-effector or changes continuously with task progress, these crops can become misaligned. We propose Seeker, a task- and state-conditioned readout that learns attention from action. Starting from frozen DINOv3 features, Seeker iteratively updates a query with gathered visual evidence, producing progression-aware ROIs solely from action supervision. The learned ROI serves as a spatial interface for RGB cropping, mask-guided background augmentation, and point-cloud filtering. In simulation and the real world, Seeker improves data efficiency and robustness over no-crop, augmentation, and action-derived crop baselines. On real robots, Seeker raises average in-domain success from the best baseline's 48.3% to 76.7% and success under lighting/background shifts from 20.0% to 60.0%.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇论文《Attention from Action, for Action: Emergent Visual Bottlenecks for Policy Learning》的分析如下：

### 1. 核心贡献总结
该论文提出了一种名为 **Seeker** 的视觉瓶颈学习框架，旨在解决具身智能中“看哪里”（视觉注意力）与“怎么做”（决策策略）之间的解耦问题。通过利用动作数据监督学习任务和状态感知的感兴趣区域（ROI），Seeker 能够无需额外标注即可在复杂环境中自动提取关键视觉信息，显著提升了策略学习的数据效率与环境鲁棒性。

### 2. 关键创新点与方法论
*   **任务与状态条件下的动态注意力机制**：不同于传统的固定裁剪（fixed crop）或基于端点（end-effector）的启发式方法，Seeker 将注意力视为一种“读出机制”（readout），利用预训练好的 DINOv3 冻结特征，通过查询更新（query update）机制，根据当前动作和任务进度动态聚焦相关区域。
*   **动作监督驱动的隐式表征**：该方法的核心在于“Attention from Action”——即通过动作轨迹反馈来监督 ROI 的产生，而非依赖昂贵的人工标注（如凝视数据或语义分割）。
*   **多模态空间接口**：学习到的 ROI 不仅用于 RGB 图像裁剪，还被转化为掩码引导的背景增强和点云过滤工具，这种跨模态的通用性是其区别于现有方法的重要特征。

### 3. 对该领域的潜在影响
*   **打破了数据标注依赖**：该研究展示了无需额外语义标签即可通过交互轨迹“涌现”出视觉注意力的潜力，这为处理大规模、长序列的机器人任务降低了数据准备成本。
*   **提升了具身智能的泛化能力**：在实验中，该方法将动态环境下的成功率从 20% 提升至 60%，证明了视觉瓶颈不仅能加速训练，还能通过过滤无关背景噪声增强策略对光照和背景变化的鲁棒性，这对于部署到真实世界的机器人至关重要。

### 4. 相关领域与潜在应用
*   **机器人操作（Manipulation）**：特别是长程、多阶段的精细操作任务，其中关键视觉线索会随时间从“端点”转移到“物体”或“环境特征”。
*   **视觉伺服与自主导航**：在需要动态调整注意焦点以处理遮挡或视角变化的场景中表现出色。
*   **端到端自动驾驶**：自动驾驶系统可借鉴此架构，在处理海量路况信息时，学习从驾驶行为中提取关键的视觉瓶颈，以提高对突发障碍物的响应能力。

### 5. 可推断的局限性
*   **依赖于预训练特征的质量**：Seeker 依赖于冻结的 DINOv3 特征，如果预训练模型在特定领域（如极其特殊的工业场景或非自然场景）表现不佳，该方法的有效性可能会大打折扣。
*   **“黑盒”注意力的可解释性**：虽然通过动作监督学习到了 ROI，但这种动态生成的 ROI 如何与人类的注意力逻辑对齐仍有待进一步探讨。
*   **序列依赖性**：由于 ROI 是根据动作监督进行迭代更新的，对于完全未见过的、非结构化的操作动作，模型是否能保持稳定的 ROI 提取（而不出现抖动或丢失焦点）可能存在挑战。

**专家简评：** 这篇论文的趣味性在于它巧妙地论证了“动作不仅是输出，还是高质量视觉特征的解释者”。通过将计算机视觉中的注意力机制与机器人动作控制进行深度融合，它为解决具身智能中最核心的“视觉瓶颈”问题提供了一条高效、实用的路径，是一项具备实际工程落地潜力的前沿工作。

**Key Findings:**

- We propose Seeker, a task- and state-conditioned readout that learns attention from action.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.13422v1)
- [arXiv](https://arxiv.org/abs/2608.13422v1)

---

<a id='2608.13147v1'></a>
## [Geometry-Grounded Unified 3D Perception for Autonomous Driving](https://arxiv.org/abs/2608.13147v1)

**Authors:** Longfei Xu, Xiaohui Wang, Zehao Huang, Han Li, Ya Yang, Naiyan Wang, Si Liu

**Published:** 2026-08-13

**Categories:** cs.CV

**Abstract:**

Camera-based autonomous driving perception requires a shared representation that preserves metric 3D structure across synchronized multi-camera streams. However, existing image-based frameworks often rely on backbones pretrained for semantic recognition, and introduce 3D geometry through downstream task-specific modules. As a result, their shared representations may fail to preserve explicit metric geometry and consistent 3D scene structure. In this paper, we present a Geometry-grounded Unified 3D Perception (GeoUP) framework that adapts the reconstruction-oriented latent of VGGT to calibrated, streaming multi-camera driving scenes. GeoUP factorizes cross-image interaction into self, temporal, and view attention to capture structurally distinct temporal and cross-view correspondences. It further injects calibration-aware raymap encodings to provide metric scale and camera geometry. The resulting geometry-grounded latent is decoded for metric depth estimation, 3D object detection, and semantic occupancy prediction, corresponding to surface-, instance-, and volume-level readouts of the same 3D scene. Through joint multi-task and multi-dataset training, GeoUP effectively leverages heterogeneous annotations and generalizes across diverse sensor configurations and perception ranges. Extensive experiments on nuScenes, Argoverse 2, Waymo, KITTI, and DDAD demonstrate that GeoUP achieves SOTA performance across detection, occupancy, and depth estimation. These results validate the effectiveness of geometry-grounded representations for unified 3D driving perception.

**Analysis:**

以下是对《Geometry-Grounded Unified 3D Perception for Autonomous Driving》一文的深度分析：

### 1. 摘要翻译
基于摄像头的自动驾驶感知需要一种能够在同步多相机流中保留度量3D结构的共享表示。然而，现有的基于图像的框架通常依赖于为语义识别预训练的骨干网络，并通过下游任务特定模块引入3D几何，这导致共享表示难以保留显式的度量几何和一致的3D场景结构。在本文中，我们提出了“几何扎根统一3D感知（GeoUP）”框架，将VGGT的重构导向潜变量适配到经过校准的流式多相机驾驶场景中。GeoUP将跨图像交互分解为自注意力、时间注意力和视图注意力，以捕捉结构上独特的时空和跨视图对应关系，并注入校准感知的射线图（raymap）编码以提供度量尺度和相机几何。由此产生的几何扎根潜变量被解码用于度量深度估计、3D目标检测和语义占据预测。通过联合多任务和多数据集训练，GeoUP有效地利用了异构标注，并推广到不同的传感器配置。在nuScenes、Argoverse 2、Waymo等基准上的实验表明，GeoUP在各项感知任务上均达到了SOTA性能。

### 2. 方法动机分析
- **驱动力**：现有的感知框架将“语义识别”作为预训练核心，通过后续模块（如BEV Lifting）“事后”修补3D几何，导致特征空间缺乏本质的物理尺度感知。
- **痛点**：缺乏多视角、跨时间的一致性表达；下游任务各自为政，无法形成统一的3D场景理解。
- **研究假设**：如果将具有强3D重构能力的“视觉几何基础模型”作为感知骨干，并在特征层注入相机校准先验，就能构建出具备度量几何感知的统一特征表示。

### 3. 方法设计详解
- **Pipeline**：
    1. **几何增强输入**：将图像Patches编码为Token，并注入基于相机内参/外参计算的6D Plücker射线编码（$R^v_t$），形成“几何感知Token”。
    2. **结构化时空注意力**：在Transformer层中，将复杂的全局注意力分解为：
        - `Attn_self`：图像内部上下文；
        - `Attn_temp`：同一相机跨帧的时间对应；
        - `Attn_view`：同一时间戳下的跨相机视图交换。
    3. **多任务解码**：通过共享的几何扎根潜变量，通过不同Head读取：
        - `Depth Head`：结合相机内参解码稠密深度；
        - `Detection Head`：使用RayDN查询机制进行3D框预测；
        - `Occupancy Head`：利用多帧特征进行时空对齐的占据空间推理。
- **算法精髓**：公式(1)通过MLP将6D Plücker射线参数投影到Token维度，使特征直接携带物理空间信息。公式(3)定义的层级注意力机制，有效降低了计算复杂度，同时保留了3D空间结构。

### 4. 方法对比分析
- **本质区别**：传统方法是“语义提取->几何对齐”，GeoUP是“几何重构->任务读取”。
- **创新贡献**：将“视觉几何基础模型”（VGGT）通过“校准感知raymap注入”和“因子化时空注意力”成功适配于自动驾驶的流式多相机场景。
- **适用场景**：需要高精度度量空间感知的自动驾驶全栈感知（深度、检测、占据）。

### 5. 实验分析
- **验证方法**：在nuScenes、Waymo、Argoverse 2等多数据集上进行多任务联合训练。
- **关键结论**：多任务联合训练显著提升了鲁棒性；Plücker射线编码对于解决绝对尺度感知至关重要。
- **局限性**：由于引入了复杂的Transformer结构进行多帧处理，单机推理延迟较高（FPS较低）。

### 6. 实用指南
- **开源**：项目主页 `https://buaa-colalab.github.io/geoup_page`。
- **关键技巧**：在多数据集训练中，保持深度尺度的归一化（统一到90m）是保证模型收敛的关键。
- **迁移可能**：该骨干模型可直接用于需要场景重建的机器人视觉、AR/VR空间定位任务。

### 7. 总结
- **核心思想**：以几何重构为骨架，通过校准信息注入构建度量一致的统一感知表示。
- **速记版Pipeline**：
    1. 注入物理射线编码，让特征“知晓”空间位置；
    2. 分解注意力，精准捕捉时间和视图关联；
    3. 联合多任务监督，让模型从不同侧面理解3D环境；
    4. 统一解码，从单一潜变量生成深度、目标与空间状态。

**Key Findings:**

- In this paper, we present a Geometry-grounded Unified 3D Perception (GeoUP) framework that adapts the reconstruction-oriented latent of VGGT to calibrated, streaming multi-camera driving scenes.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.13147v1)
- [arXiv](https://arxiv.org/abs/2608.13147v1)

---

<a id='2608.13113v1'></a>
## [EgoMonth: A Month-Level Egocentric Video Benchmark for Long-Term Spatiotemporal Memory](https://arxiv.org/abs/2608.13113v1)

**Authors:** Weitao Chen, Hu Jiaxin, Xie Tianyidan, Yang Li, Yuyi Qian, Banghao Xu, Ziheng Tang, Shenyi Wang, Mingyue Yu, Duo Li, Jiacheng Shi, Gao Wang, Zhan Xu, Zhicheng Qiu, Xuanfu Li, Jian Yang, Lanjun Wang, Zili Yi

**Published:** 2026-08-13

**Categories:** cs.CV, cs.AI

**Abstract:**

Recent advances in Multimodal Large Language Models (MLLMs) have led to substantial progress in video understanding, accompanied by a growing number of long video benchmarks. However, existing benchmarks rely predominantly on web-sourced videos that lack inter-clip spatiotemporal continuity, making it difficult to assess whether models can maintain consistent memory across days or weeks of real-world experience. We introduce EgoMonth, the first month-level egocentric video understanding benchmark. EgoMonth comprises over 300 hours of first-person daily-life recordings from 20 participants spanning 20 to 120 days, paired with 1,443 human-crafted multiple-choice question-answer pairs. We design a cognitively grounded 14-task evaluation framework organized into three hierarchical cognitive levels: Schema Consolidation, Episodic Indexing, and Cascading Reasoning. Evaluation of state-of-the-art open-source and closed-source MLLMs reveals that even the best-performing model, Gemini 2.5 Pro, achieves only 71.8% macro-average accuracy, remaining 22.4 percentage points below the corrected human baseline of 94.2%. Several models perform near or below the 25% chance level on tasks such as Route Reasoning, Cross-view Spatial Reasoning, and Direction Judgement, while even the strongest closed-source model remains substantially below human performance. These results indicate that current MLLMs function as lossy summarizers rather than faithful memorizers, highlighting the need for architectures with genuine long-term spatiotemporal memory.

**Analysis:**

### 1. 摘要翻译
多模态大语言模型（MLLMs）在视频理解领域取得了显著进展，且长视频基准测试数量日益增长。然而，现有基准测试主要依赖缺乏跨片段时空连续性的网络视频，这使得评估模型在跨天或跨周的真实生活经历中是否具备一致的长期记忆变得困难。本文介绍了EgoMonth，这是首个月份级的自中心视频理解基准测试。EgoMonth包含来自20名参与者、跨度为20至120天的超过300小时的真实第一人称记录，并配有1,443个手工制作的多选题问答（QA）对。我们设计了一个认知驱动的14项任务评估框架，分为三个层级：模式巩固、情景索引和级联推理。对先进开源和闭源MLLM的评估显示，表现最好的模型Gemini 2.5 Pro的宏平均准确率仅为71.8%，比人类基准（94.2%）低22.4个百分点。

### 2. 方法动机分析
*   **驱动力**：旨在构建一个能够评估人工智能在处理真实世界、长期、连续、高冗余自中心视频数据时，是否存在“真实”长时记忆（即能否跨天/跨周保持长期一致性）的评价指标。
*   **痛点**：现有视频基准测试多基于电影或网络短片，缺乏环境持久性与行为习惯的连贯性，导致模型在处理“稀疏事件记忆”和“复杂时空跨度推理”时难以获得准确评测。
*   **研究假设**：通过引入认知科学中的分层结构（Schema Consolidation, Episodic Indexing, Cascading Reasoning），可以将长时记忆能力解耦为感知巩固与逻辑推理，从而精准诊断MLLM作为“有损压缩器”而非“忠实记忆者”的瓶颈。

### 3. 方法设计详解
*   **核心 Pipeline**：
    1.  **数据采集与隐私化**：采集20名志愿者的长达数月的真实第一人称视频（>300小时），通过Grounding DINO 1.5和SAM 2进行强力的隐私自动标注与掩码。
    2.  **认知任务体系设计**：将评估分为三层：
        *   **模式巩固（Schema Consolidation）**：评估对习惯、性格等长期规律的归纳能力。
        *   **情景索引（Episodic Indexing）**：评估对特定时间、空间位置、物体状态的检索能力。
        *   **级联推理（Cascading Reasoning）**：评估跨天、跨环境、多步骤的依赖性推理（如路径重构、事件计数）。
    3.  **多选QA构建**：手工构建1,443个高质量QA对，通过“先浏览、后命题、多人交叉验证”的方式确保答案的唯一性与证据的可追溯性。

### 4. 方法对比分析
*   **本质区别**：与Ego4D等侧重于动作识别或短片段理解不同，EgoMonth强调“月份尺度（Month-level）”的持久性，其QA设计强制要求模型跨多个独立视频文件整合记忆证据。
*   **创新贡献**：首次将认知科学的记忆模型引入视频Benchmark设计，提出了针对“跨天时空关联”的评估任务。
*   **适用场景**：适用于评估智能体（Agent）在长期个人助手应用中的长期上下文记忆与行为理解能力。

### 5. 实验分析
*   **关键结果**：目前最强模型Gemini 2.5 Pro在级联推理任务上仍显著落后于人类。
*   **主要优势**：不仅是一个数据集，更是一个诊断框架，揭示了模型性能下降并非因为缺乏计算力，而是因为缺乏“结构化时空表示”。
*   **主要局限**：模型容易因时空冗余而陷入干扰，难以实现跨片段的有效对齐。

### 6. 实用指南
*   **开源情况**：数据集已发布，采用分级访问控制以保障隐私。
*   **关键注意**：在处理长时自中心视频时，单纯增加帧采样密度（Frame Density）并不能保证提升准确率，关键在于对关键帧的选择和对跨片段的逻辑索引。
*   **迁移建议**：文中提出的三级认知评估架构（Schema-Indexing-Reasoning）可用于构建其他领域的长时视频理解任务，如医学诊断或工业流水线监控。

### 7. 总结
*   **核心思想**：通过认知分层设计，破解MLLM在真实长时视频中的记忆崩溃瓶颈。
*   **速记版Pipeline**：
    1. 收集长期跨度的真实第一人称原始视频；
    2. 隐私识别与自动掩码处理；
    3. 设计三级认知难度（模式、索引、推理）任务；
    4. 引入跨视频检索机制构造QA对；
    5. 通过多指标准确率诊断模型记忆缺陷。

**Key Findings:**

- We introduce EgoMonth, the first month-level egocentric video understanding benchmark.
- Evaluation of state-of-the-art open-source and closed-source MLLMs reveals that even the best-performing model, Gemini 2.5 Pro, achieves only 71.8% macro-average accuracy, remaining 22.4 percentage points below the corrected human baseline of 94.2%.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.13113v1)
- [arXiv](https://arxiv.org/abs/2608.13113v1)

---

<a id='2608.13045v1'></a>
## [P2Fusion: Prompt-based Progressive Infrared-Visible Image Fusion via Dual-Prior Distillation](https://arxiv.org/abs/2608.13045v1)

**Authors:** Yi Shi, Huichao Xie, Yuqing Wang, Mingyu Wang, Kaihui Yang, Yu Liu, Ruitao Lu, Lizhe Li, Junwei Han, Dingwen Zhang

**Published:** 2026-08-13

**Categories:** cs.CV

**Abstract:**

Infrared-visible image fusion (IVIF) is pivotal for multimodal perception, yet reconciling the inherent information disparity between thermal and textural features remains a fundamental challenge. Existing prior-guided methods often rely on static constraints that induce optimization conflicts or utilize extrinsic semantic priors from large-scale foundation models (e.g., CLIP/DINO), which frequently fail to exploit the intrinsic modality characteristics essential for high-fidelity fusion. To address these issues, we propose P2Fusion, a prior-guided distillation-based framework that reformulates IVIF via dual intrinsic prompts. Instead of imposing hard-coded penalties, we distill image-intrinsic priors, thermal saliency and spatial quality, into learnable dynamic regulators. Specifically, a Teach-to-Fuse mechanism provides dual-granularity progressive guidance, coupled with a Gated Dynamic Expert Recalibration (GDER) module for decoupled feature refinement. This design enables the network to adaptively mediate modal competition through expert specialization. Extensive experiments demonstrate that P2Fusion achieves state-of-the-art performance across five mainstream datasets. Notably, our framework demonstrates consistent performance advantages in fusion quality, achieving state-of-the-art results in 14 out of 20 key evaluation metrics across 5 benchmarks. Furthermore, it effectively contributes to the robustness of downstream perception, such as +3.2% mAP on MSRS, +0.5% mAP on M3FD and +0.9% mAP on DroneVehicle for object detection. Our code will be available at https://github.com/YiShi99/P2Fusion

**Analysis:**

### 1. 摘要翻译
红外-可见光图像融合（IVIF）对于多模态感知至关重要，但如何调和热辐射与纹理特征之间的信息差异仍是一项基础挑战。现有的先验引导方法往往依赖导致优化冲突的静态约束，或利用来自大规模基础模型（如CLIP/DINO）的外在语义先验，这往往无法挖掘对高保真融合至关重要的内在模态特性。为了解决这些问题，我们提出了P²Fusion，一个通过双重内在提示（dual intrinsic prompts）重构IVIF的先验引导蒸馏框架。我们没有使用硬编码惩罚，而是将图像内在先验——热红外显著性和空间质量——蒸馏为可学习的动态调节器。具体而言，“Teach-to-Fuse”机制提供了双粒度渐进式引导，并结合门控动态专家重校准（GDER）模块进行解耦特征细化。该设计使网络能够通过专家专门化自适应地调节模态竞争。大量实验表明，P²Fusion在五个主流数据集上实现了最先进（SOTA）的性能。

---

### 2. 方法动机分析
*   **驱动力**：解决现有先验引导方法中“静态约束导致优化冲突”和“外在语义先验与像素级融合任务之间存在粒度不匹配”的问题，回归到挖掘模态内在特性。
*   **现有方法痛点**：
    1.  **硬约束限制**：将先验视为静态惩罚项，迫使网络牺牲纹理细节以拟合先验边界。
    2.  **多任务冲突**：联合优化下游任务导致精度与背景细节保护之间的平衡失调。
    3.  **粒度不匹配**：外部语义特征（如CLIP）对于重建底层的纹理像素缺乏精确引导。
*   **研究假设**：先验知识应以“软的、动态的”提示（prompt）形式参与融合过程，而非作为硬性的空间约束。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **双教师蒸馏**：利用SegFormer-B2生成的“红外显著性教师”和基于BRISQUE的“可见光质量教师”，将红外目标定位和可见光纹理可靠性作为动态提示输入。
    2.  **跨模态交互**：采用交叉注意力（Cross-Attention）机制进行初步的特征调制，初步融合两个模态的信息。
    3.  **GDER精细化**：核心模块，通过门控机制将特征分配给两个专家：
        *   **Prior-Responsive Modality Expert (Emod)**：利用动态提示强化显著信息（如热辐射目标）。
        *   **Prior-Agnostic Attention Expert (Eatt)**：利用CBAM捕获长距离依赖和精细纹理。
    4.  **特征重构**：将专家输出加权融合，最后通过解码器重构出最终图像。
*   **算法解释**：核心公式 $F'' = F' + w_1 \cdot E_{mod}(F') + w_2 \cdot E_{att}(F')$。 gating network 计算权重 $w$，实现了对冲突特征的动态解耦。

---

### 4. 方法对比分析
*   **本质区别**：从“规则约束”转向“动态提示引导”，强调内在属性而非外部语义。
*   **创新贡献**：引入GDER模块，实现了基于专家专门化的自适应特征修正，有效规避了多模态融合中常见的特征耦合问题。
*   **适用场景**：极端光照、烟雾遮挡及超视距等需高动态范围融合的场景。

---

### 5. 实验分析
*   **关键结论**：在MSRS、M3FD等5个基准上，P²Fusion在20项指标中获得14项SOTA；在下游目标检测任务中，mAP平均提升显著（如MSRS提升3.2%）。
*   **优势**：在保持热辐射显著目标的同时，极大改善了背景纹理的恢复。
*   **局限**：对教师模型的性能有一定依赖（虽然采用了动态提示，但仍需训练教师）。

---

### 6. 实用指南
*   **开源地址**：[https://github.com/YiShi99/P2Fusion](https://github.com/YiShi99/P2Fusion)
*   **实现细节**：GDER模块中的 $N$ 次迭代精炼是提升质量的关键；损失函数权重的合理设置（$\lambda_1-\lambda_5$）对于平衡热辐射与纹理至关重要。
*   **迁移可能**：该框架的“动态提示+双专家纠偏”架构可直接迁移至多源传感器融合（如LiDAR+RGB）。

---

### 7. 总结
*   **核心思想**：利用双重内在提示蒸馏与解耦专家模型，实现自适应动态特征融合。
*   **速记版pipeline**：
    1. 训练双教师模型生成提示图；
    2. 执行跨模态交叉注意力融合；
    3. 通过专家模型进行解耦特征校准；
    4. 重构最终融合图像。

**Key Findings:**

- To address these issues, we propose P2Fusion, a prior-guided distillation-based framework that reformulates IVIF via dual intrinsic prompts.
- Extensive experiments demonstrate that P2Fusion achieves state-of-the-art performance across five mainstream datasets.
- Notably, our framework demonstrates consistent performance advantages in fusion quality, achieving state-of-the-art results in 14 out of 20 key evaluation metrics across 5 benchmarks.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.13045v1)
- [arXiv](https://arxiv.org/abs/2608.13045v1)

---

<a id='2608.13028v1'></a>
## [RGB-D Video Generation for Improving Human-to-Robot Object Handover Prediction](https://arxiv.org/abs/2608.13028v1)

**Authors:** Tianyu Sun, Zhoujie Fu, Zihui Gao, Bang Zhang, Guosheng Lin

**Published:** 2026-08-13

**Categories:** cs.CV, cs.RO

**Abstract:**

Human-to-robot (H2R) object handover is a fundamental capability for human-robot collaboration, yet progress is hindered by the scarcity of large-scale, human-centric datasets and the significant sim-to-real gap. To address these challenges, we introduce Hand2Bot, an RGB-D video dataset that provides rich contextual information such as body posture and facial expressions, specifically collected for handover scenarios with real-world noise patterns. We further propose PassGen, a generative pipeline that leverages stable video diffusion and an Intention-Aware Temporal Face Encoder to synthesize realistic handover sequences while ensuring hand-object consistency. To bridge the sim-to-real gap, we implement a morphology-based depth editing strategy that replicates realistic sensor noise found in physical depth maps. Experimental evaluations demonstrate that our framework achieves high intention identification accuracy and low false trigger rates in both ablation studies and real-world deployment on a physical robot platform. Our results confirm that training on PassGen allows for robust zero-shot transfer and earlier intention anticipation compared to traditional hand-centric baselines, effectively enabling socially aware robotic behavior in shared workspaces.

**Analysis:**

这是一份针对论文《RGB-D Video Generation for Improving Human-to-Robot Object Handover Prediction》的深度分析报告。

---

### 1. 摘要翻译
人机（H2R）对象递交是人机协作的核心能力，但受限于大规模人机数据集的匮乏及严重的“仿真到现实”（sim-to-real）鸿沟。为解决此问题，我们引入了 **Hand2Bot**，这是一个专门收集的包含真实世界噪声模式且具有丰富情境（如身体姿态、面部表情）的RGB-D视频数据集。我们进一步提出了 **PassGen**，这是一种利用稳定视频扩散模型（Stable Video Diffusion）和意图感知时间面部编码器（Intention-Aware Temporal Face Encoder）的生成管线，在确保手-物一致性的同时合成逼真的递交序列。为弥合sim-to-real鸿沟，我们实现了基于形态学的深度编辑策略，用以复制物理深度图中的真实传感器噪声。实验证明，该框架在消融实验和物理机器人平台的真实部署中均实现了高意图识别准确率和低误触发率。结果证实，基于PassGen的训练支持鲁棒的零样本迁移和更早的意图预期，从而在共享工作空间中实现社会感知型的机器人行为。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决现有H2R数据集缺乏“全身体感知”和“真实传感器噪声”的问题，从而提升机器人对人类意图的理解力，使其在复杂环境中做出更平滑的反应。
*   **现有方法痛点**：
    1. 现有数据多为合成，缺乏真实传感器（如RealSense L515）的边缘散射等噪声，导致训练模型无法适配现实环境。
    2. 现有基准偏重手部细节，忽略了面部表情、注视方向等对于预测人类意图至关重要的全局社会信号。
*   **研究假设**：通过引入意图感知的人体全身体（RGB）信息以及符合物理特性的深度噪声模拟，模型能更精准地识别递交行为，并有效过滤环境干扰。

---

### 3. 方法设计详解
**流程总结：**
1. **Pose-Guided RGB视频生成（Stage I）**：基于SVD架构，引入外观编码器（提取身份）和PoseNet（提取骨架）。核心创新是**Intention-Aware TFE（时间面部编码器）**，它将面部视为高频语义流而非静态组件，通过ArcFace提取嵌入并注入U-Net中，以保留递交过程中的视觉专注焦点。
2. **形态学深度视频生成（Stage II）**：使用DepthCrafter获得平滑的初始深度图，随后记录真实传感器的噪声分布，采用**边缘腐蚀（Edge Erosion）**算法，根据边界向量调整像素高度，重构符合真实传感器特性的“空洞”噪声，实现Sim-to-Real的无缝对接。

---

### 4. 方法对比分析
*   **本质区别**：传统模型侧重于“动作复刻”，而PassGen侧重于“意图表征”，将面部 gaze/micro-expression 等社会线索与动作解耦并显式注入。
*   **创新贡献**：提出了一种结合形态学噪声模拟与社会意图感知（TFE）的生成管线，直接从生成层面解决了机器人视觉感知的Sim-to-Real鸿沟。
*   **适用场景**：适用于人机递交任务，尤其是需要机器人“提前感知”人类需求、避免在环境干扰下产生误动作的协作场景。

---

### 5. 实验分析
*   **验证方法**：使用Hand2Bot-Real真实数据集对生成质量进行评测；在物理UR5e机器人平台上进行递交实验。
*   **关键结果**：在PassGen加持下，机器人对未见过（Unseen）物体的意图识别成功率提升至7/10，且误触发率（FTR）显著下降至13.6%。
*   **主要优势**：不仅提升了生成图像的视觉真实度，更通过意图门控机制（IG）大幅降低了背景环境对机器人决策的干扰。
*   **主要局限**：对“不规则几何形状”物体在 grasp 完成阶段仍存在物理抓取困难，本质原因是生成策略与机器人动力学控制仍存在微小的物理匹配差异。

---

### 6. 实用指南
*   **开源与复现**：作者通过LoRA方案对SVD进行微调。建议关注其对RealSense L515传感器噪声分布的采样记录方式，这是模型泛化到物理环境的核心。
*   **实现细节**：在推理阶段，意图门控机制设置权重 $w_g = 1.0, w_v = 2.5[s/m]$ 以及阈值 $\tau = 0.8$ 是实现平滑切换的关键。
*   **迁移可能**：该架构可以轻松迁移到其他需要人机交互感知（如辅助喂食、家庭管家机器人）的视觉控制任务中。

---

### 7. 总结
*   **核心思想**：通过意图感知的视频生成和真实噪声建模，将人的社会性意图与物理动作协同对齐。
*   **速记版pipeline**：
    1. **特征提取**：利用ArcFace捕捉注视意图，PoseNet获取骨架动作。
    2. **扩散生成**：在SVD骨干网中显式注入意图信息，生成RGB序列。
    3. **深度渲染**：对合成深度图进行基于形态学的传感器噪声侵蚀。
    4. **意图门控**：综合面部关注度和目标移动速度，判断是否开启抓取指令。

**Key Findings:**

- To address these challenges, we introduce Hand2Bot, an RGB-D video dataset that provides rich contextual information such as body posture and facial expressions, specifically collected for handover scenarios with real-world noise patterns.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.13028v1)
- [arXiv](https://arxiv.org/abs/2608.13028v1)

---

<a id='2608.12860v1'></a>
## [HumanoidVLN: A Physics-Grounded Simulator and Benchmark for Vision-Language Navigation Across Diverse Humanoid Embodiments](https://arxiv.org/abs/2608.12860v1)

**Authors:** Quan-Dung Pham, Anh Dao, The-Anh Nguyen, Minh Nguyen-Dinh, Phuong Nam Dang, Tri Pham, Hung Tran, Bach Dao, Tuyen P. Le, Truong Nguyen, Quan Nguyen

**Published:** 2026-08-13

**Categories:** cs.RO

**Abstract:**

Vision-Language Navigation (VLN) for humanoid robots poses challenges existing benchmarks fail to address: bipedal locomotion imposes physical constraints absent from wheeled agents, humanoid morphologies vary across platforms, and egocentric observations are distorted by locomotion-induced camera dynamics. We present HumanoidVLN, a physics-grounded simulator and benchmark for VLN across diverse humanoid embodiments. Built on NVIDIA Isaac Sim, our platform supports an extensible set of humanoid configurations, demonstrated on four robots (Unitree G1, Unitree H1, Internal-A, Internal-B) spanning 10-12 lower-body DoF and heights from 1.17m to 1.80m, via a hierarchical control stack combining a reinforcement learning locomotion policy with interchangeable PD or MPC path trackers. New robots and VLN models integrate with minimal effort; we demonstrate compatibility with NaVILA, DualVLN, StreamVLN, and JanusVLN. Environments are drawn from artist-designed scenes and 3D Gaussian Splatting reconstructions, filtered for navigable areas exceeding 100 square meters. Instructions are generated by a dual generator-reviewer plus paraphraser multi-agent pipeline with human-in-the-loop verification, yielding 933 collision-aware reference episodes, each paired with one fine-grained instruction and three coarse-grained stylistic variants (formal, natural, casual). Across four models and four embodiments, JanusVLN achieves the highest mean success rate of 43.55% and nDTW of 48.38. In a 20-episode sim-to-real pilot with DualVLN and the Unitree G1, navigation errors correlate strongly (r=0.935), with a mean absolute difference of 0.68m and mean trajectory similarity of 0.782 (+/-0.188) nDTW. These results highlight the interaction between VLN models, controllers, and humanoid embodiments under physical execution. Code, benchmark, and data will be released upon acceptance at https://humanoid-vln.github.io/.

**Analysis:**

### 1. 摘要翻译
本文提出了HumanoidVLN，一个用于跨多种人形机器人形态的视觉语言导航（VLN）物理仿真平台与基准测试。现有基准测试忽略了双足运动的物理约束、人形形态的差异以及运动诱发的视觉失真。HumanoidVLN基于NVIDIA Isaac Sim，支持包括Unitree G1、H1等多种人形机器人配置，通过分层控制栈（RL运动策略+PD/MPC路径跟踪）实现了物理真实的导航。此外，作者构建了87个高保真3D Gaussian Splatting（3DGS）环境，并提出了一种基于多智能体标注（MAA）的方法生成933个碰撞感知参考片段。实验表明，该平台能有效评估模型在物理执行下的导航能力，并弥合了仿真与现实之间的差距。

### 2. 方法动机分析
*   **驱动力**：旨在填补现有的VLN基准测试中对“机器人本体物理限制”的忽视，实现真正具备物理落地潜力的导航评估。
*   **现有方法痛点**：
    *   **运动仿真过于理想化**：传统方法通过运动学传送（Teleportation）简化导航，忽略了双足行走中的CoM动态、步态不稳定及复杂地形交互。
    *   **环境缺乏 navigability 筛选**：现有场景未针对双足机器人进行过滤，导致地形拓扑极其复杂，人形机器人难以完成任务。
    *   **数据源与真实观测不符**：现有标注基于理想视觉流，忽略了双足行走产生的相机抖动和光照变化，导致模型在Sim2Real中失效。
*   **研究假设**：通过在分层控制栈下强制执行物理运动，并对环境进行 navigability 筛选，可以构建一个更能真实反映机器人导航能力的基准，从而有效缩小Sim2Real差距。

### 3. 方法设计详解
*   **流程总结**：
    1.  **形态多样化仿真**：基于Isaac Sim构建支持多种人形机器人的平台。
    2.  **分层控制架构**：底层为RL运动策略（处理平衡与关节控制），高层为PD/MPC跟踪器（处理路径规划），确保每一步路径都是物理可执行的。
    3.  **场景Curated构建**：利用3DGS重建真实环境，通过遍历性（Navigability）筛选（>100m²可通行区域）构建高质量环境。
    4.  **MAA数据标注**：利用“双生成器+对比+审核员+人类验证”的闭环流程，生成具备物理一致性的导航指令。
*   **核心模块**：
    *   **MAA标注框架**：不同于传统的单步LLM标注，该框架让两个VLM独立生成路径图，通过对比发现冲突，再由第三方Reviewer进行验证，最后人类介入，极大提升了空间指令的可靠性。
    *   **Fall Detection策略**：不仅关注成功率，还显式定义了动态跌倒（T1）、持续坍塌（T2）等指标，用于评估运动稳定性。

### 4. 方法对比分析
*   **本质区别**：从“路径规划层”下沉到了“全身动力学层”，确保导航指令不仅是逻辑上的可行，更是机器人躯体动力学层面的可行。
*   **创新贡献**：提出了一种结合3DGS重构与多智能体协同标注的流水线，使得数据集具有极高的视觉真实感和空间锚定精度。
*   **适用场景**：适用于人形机器人导航的研究，特别是对运动稳定性、Sim2Real迁移性有严格要求的任务。

### 5. 实验分析（精简版）
*   **关键结论**：JanusVLN模型在物理执行下表现最好（SR 43.55%），且DualVLN在运动稳定性（Fall Rate）方面表现最稳。
*   **主要优势**：提供了物理级别的轨迹一致性验证，且Sim2Real相关性高达0.935（Pearson r）。
*   **主要局限**：对计算资源要求较高（全物理仿真），且目前的场景多样性仍受限于重建数据的规模。

### 6. 实用指南
*   **开源情况**：已承诺在相关网站发布代码与数据。
*   **实现细节**：建议在Isaac Sim 5.1上使用gsplat库进行重建；关键在于确保所导入机器人URDF的动力学参数与实际一致，否则会导致严重的运动失稳。
*   **迁移可能**：该MAA标注框架可以轻松迁移至其他具身智能体（如四足机器人、轮式移动操作平台）的指令生成任务中。

### 7. 总结
*   **核心思想**：将双足运动动力学纳入导航基准，实现物理真值驱动的VLN评估。
*   **速记版pipeline**：
    1. 搭建多形态人形机器人仿真模拟器；
    2. 筛选并构建高可通行性的3D真实场景；
    3. 通过多智能体对比标注生成高质路径指令；
    4. 结合物理层控制器进行零样本导航评估。

**Key Findings:**

- We present HumanoidVLN, a physics-grounded simulator and benchmark for VLN across diverse humanoid embodiments.
- New robots and VLN models integrate with minimal effort; we demonstrate compatibility with NaVILA, DualVLN, StreamVLN, and JanusVLN.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.12860v1)
- [arXiv](https://arxiv.org/abs/2608.12860v1)

---

<a id='2608.13556v1'></a>
## [V-RAE: Rethinking Video Latent Spaces for Generation](https://arxiv.org/abs/2608.13556v1)

**Authors:** Minghui Guo, Shengqiong Wu, Hao Fei

**Published:** 2026-08-13

**Categories:** cs.CV

**Abstract:**

Latent video generation relies on autoencoders to define a compact space in which generative models operate. Although video autoencoder architectures have evolved substantially, their latent spaces are still optimized primarily for pixel-level reconstruction and provide limited high-level semantic organization. A reconstruction-optimal latent space, however, need not be well suited to generative modeling. We propose V-RAE, a video representation autoencoder that builds compact generative latents on top of frozen vision foundation model representations. A lightweight temporal pooling module removes temporal redundancy while preserving semantic structure, and a video decoder reconstructs continuous motion from the compressed features. We evaluate V-RAE with four representative frozen encoders on video reconstruction, semantic probing, and class-conditional generation. V-RAE achieves 2.13 rFVD on K600, outperforming all evaluated large-scale pretrained video VAEs. Its latents retain substantially more semantic information than conventional video tokenizer latents. Under matched generation settings, our best variant achieves gFVD scores of 117.86 and 19.16 on UCF101 and K600, respectively, while converging up to 6x faster}. We further show that reconstruction quality alone is insufficient to characterize generative utility and introduce tFVD, a temporal-coherence diagnostic that correlates more reliably with downstream generation quality. Beyond video generation, V-RAE also improves future video prediction on Cityscapes over the Wan 2.2 VAE latent space under matched prediction settings. Taken together, the experiments show that frozen semantic representations can support video reconstruction, generation, and predictive modeling. The project page: https://v-rae.github.io/.

**Analysis:**

以下是对论文《V-RAE: Rethinking Video Latent Spaces for Generation》的深度分析：

### 1. 摘要翻译
潜在视频生成依赖于自动编码器构建的紧凑空间。尽管视频自动编码器已大幅演进，但其潜在空间主要优化像素级重构，缺乏高层语义组织。本文提出V-RAE，一种构建在冻结视觉基础模型（VFM）表征之上的视频表征自动编码器。它引入轻量级时间池化模块去除时间冗余并保留语义结构，利用视频解码器重构连续动作。实验表明，V-RAE在K600上达到2.13 rFVD，大幅超越现有预训练视频VAE，且潜在空间语义丰富。在匹配设置下，其生成收敛速度快达6倍。此外，作者提出tFVD指标，证明重构质量并非生成效能的充分条件，并展示了V-RAE在视频预测任务上的优势。

### 2. 方法动机分析
- **核心痛点**：现有主流视频VAE均以“像素级重构（PSNR/SSIM）”为首要目标，忽略了生成模型真正需要的“语义组织性”和“分布平滑性”。
- **驱动力**：利用冻结的、强大的视觉基础模型（如DINOv3）提取语义特征作为潜在空间，而非从零开始训练一个视频表征。
- **核心直觉**：语义丰富的特征天然更利于生成模型学习，重构任务应作为辅助而非主导，通过高效的压缩机制平衡语义保留与计算量。

### 3. 方法设计详解
- **Pipeline**：
    1. **编码阶段**：利用冻结的预训练VFM（处理单帧或原生视频块）提取密集特征。
    2. **时间池化（Temporal Pooling）**：核心创新模块，采用**时间注意力机制**，将冗余的密集时间特征通过可学习的Query和注意力权重压缩至目标长度，而非简单的固定下采样。
    3. **解码阶段**：轻量级Transformer解码器，结合**3D RoPE位置编码**，将压缩后的Chunk特征映射回RGB空间，实现帧的持续生成。
- **关键算法**：
    - **时间注意力池化**：通过公式 $z_{t,p} = \text{Norm}(\text{Concat}(u_m))$，在不破坏空间语义的前提下，实现内容自适应的时间压缩。
    - **tFVD指标**：通过对潜空间进行线性插值并解码，考察潜空间的“时间平滑性”。如果插值解码效果差，说明潜空间流形高度不连续，不利于生成。

### 4. 方法对比分析
- **本质区别**：不重构像素，而是“利用已有的语义知识（冻结VFM）”重构特征，且通过tFVD指导潜空间设计。
- **创新点**：
    1. 提出了以生成为导向的视频压缩框架，而非以重构为导向。
    2. 定义了tFVD，揭示了“重构好≠生成好”的现象。
- **适用场景**：适用于资源有限但追求高质量生成效果的视频生成任务，特别是当需要利用已有强大预训练模型作为基石时。

### 5. 实验分析
- **关键结果**：在K600上rFVD达到2.13，并在保持同等生成质量下，收敛速度较基线模型（如Wan2.2 VAE）提升至多6倍。
- **主要优势**：极强的语义保留能力（线性探针准确率高）、生成过程更易优化（收敛快）、潜空间更平滑（tFVD指标优异）。
- **主要局限**：在PSNR、SSIM等纯重构指标上表现平平（符合预期，因其非像素级重构导向），且尚未在超大规模、超长序列生成任务中验证。

### 6. 实用指南
- **开源情况**：项目主页已提供[https://v-rae.github.io/](https://v-rae.github.io/)。
- **迁移建议**：若要迁移至其他任务，重点应在于“选择最匹配的VFM作为基石”以及“微调时间池化模块”。无需对原VFM进行重训，只需调整Pooler和Decoder架构。
- **注意事项**：实验显示mean pooling在语义保持上最好，但压缩性能差；应始终优先选择“时间注意力池化”以达到性能平衡。

### 7. 总结
- **核心思想**：冻结语义特征作为生成基础，通过内容自适应压缩实现高效视频生成。
- **速记版Pipeline**：
    1. 冻结VFM特征提取。
    2. 时间注意力池化压缩特征。
    3. 3D-RoPE视频Transformer解码。
    4. tFVD评估潜空间生成潜力。

**Key Findings:**

- We propose V-RAE, a video representation autoencoder that builds compact generative latents on top of frozen vision foundation model representations.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.13556v1)
- [arXiv](https://arxiv.org/abs/2608.13556v1)

---

