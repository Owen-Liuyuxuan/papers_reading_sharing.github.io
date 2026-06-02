time: 20260602

# Arxiv Computer Vision Papers - 2026-06-02

## Executive Summary

# 每日报告执行摘要：计算机视觉前沿（2026-06-01）

## 一、主要主题与趋势

本日论文覆盖了计算机视觉的多个活跃方向，呈现出三大核心趋势：

1. **生成式模型的深化与评测**：多篇论文关注视频生成（LongLive-RAG）、3D场景合成（4D LiDAR）、人体建模（HumanNOVA）等任务，同时出现对低层视觉基准的重新审视（LL-Bench），表明领域正从“能否生成”转向“如何可靠评估与可控生成”。
2. **表征学习的理论化与结构化**：JEPA训练的正则化（VISReg）、3D几何数据的测地线引导学习（Geodesic-Guided）、以及功能理解中的可供性基础模型（AFUN），反映出对特征不变性、几何拓扑和语义功能结构的高度重视。
3. **具身智能与感知-行动闭环**：机器人世界模型（RoboDream）、策略驱动的注视成像（Foveated Imaging）以及视频推理的在线优化（VLMs as Teachers），凸显了视觉系统与决策、控制、交互的融合趋势。

## 二、重要性与创新性突出的论文

- **VISReg**（论文1）：提出方差-不变性-素描正则化，为JEPA自监督学习提供新的理论工具，可能影响视觉基础模型的高效训练范式。
- **RoboDream**（论文4）：构建组合式世界模型以大规模合成机器人训练数据，直接服务于具身智能的数据瓶颈问题，实用价值显著。
- **HumanNOVA**（论文5）：实现从单张照片快速生成真实感三维人体化身，强调普适性与速度，对数字人、虚拟现实等领域有突破性意义。
- **LL-Bench**（论文9）：系统反思大规模生成模型时代低层视觉评估标准的缺失，为建立新基准提供关键思考，推动领域规范化。

## 三、新兴研究方向与技术

- **不确定性感知的4D场景合成**（论文10）：提出“非所有点都同等重要”的思想，将不确定性建模引入LiDAR场景生成，预示着点云数据中结构化噪声处理的精细化趋势。
- **测地线引导的3D表征**（论文3）：从外在坐标转向内在几何（测地距离），为无监督3D特征学习开辟新路径。
- **自适应测试时优化**（论文7）：利用大语言模型作为教师，在推理阶段动态优化视频理解，代表了视觉-语言模型“推理时学习”的新范式。
- **长视频生成的检索增强框架**（论文8）：结合RAG技术解决长视频生成的时序一致性与上下文连贯性问题，是生成模型与信息检索交叉的重要探索。

## 四、建议全文阅读的论文（优先级排序）

1. **VISReg** — 对自监督学习理论有深刻贡献，适合从事表征学习的研究者。
2. **RoboDream** — 机器人学习与合成数据的前沿工作，对具身智能从业者极具启发。
3. **HumanNOVA** — 快速真实感人体建模，涉及图形学、渲染与生成，应用面广。
4. **LL-Bench** — 评估基准的批判性反思，所有从事生成模型评测的研究者都应了解。
5. **Policy-based Foveated Imaging**（论文6）— 将强化学习与生物启发的注视机制结合，新颖性强，适合关注高效感知系统的人士。

总结而言，本日论文标志着计算机视觉正从以大模型规模扩展为主，转向更加注重结构性先验、任务特定优化、以及与实际机器人/交互系统的闭环整合。

---

## Table of Contents

1. [VISReg: Variance-Invariance-Sketching Regularization for JEPA training](#2606.02572v1)
2. [AFUN: Towards an Affordance Foundation Model for Functionality Understanding](#2606.02551v1)
3. [From Extrinsic to Intrinsic: Geodesic-Guided Representation Learning for 3D Geometric Data](#2606.02268v1)
4. [RoboDream: Compositional World Models for Scalable Robot Data Synthesis](#2606.02577v1)
5. [HumanNOVA: Photorealistic, Universal and Rapid 3D Human Avatar Modeling from a Single Image](#2606.02573v1)
6. [Policy-based Foveated Imaging and Perception](#2606.02565v1)
7. [VLMs are Good Teachers for Video Reasoning via Adaptive Test-Time Optimization](#2606.02564v1)
8. [LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation](#2606.02553v1)
9. [LL-Bench: Rethinking Low-Level Vision Evaluation in the Era of Large-Scale Generative Models](#2606.02535v1)
10. [Not All Points Are Equal: Uncertainty-Aware 4D LiDAR Scene Synthesis](#2606.02510v1)

---

## Papers

<a id='2606.02572v1'></a>
## [VISReg: Variance-Invariance-Sketching Regularization for JEPA training](https://arxiv.org/abs/2606.02572v1)

**Authors:** Haiyu Wu, Randall Balestriero, Morgan Levine

**Published:** 2026-06-01

**Categories:** cs.CV

**Abstract:**

Self-supervised learning methods prevent embedding collapse via modeling heuristics or explicit regularization of the embedding space. Among the latter, VICReg decomposes regularization into variance and covariance objectives, offering flexibility and interpretability. However, covariance captures only second-order statistics -- encouraging decorrelation but failing to enforce the full distributional shape needed for stable training. Sketching-based methods such as SIGReg address this by aligning embeddings to an isotropic Gaussian, but lack flexibility and suffer from vanishing gradients under collapse. We propose Variance-Invariance-Sketching Regularization (VISReg), which replaces covariance with a Sliced-Wasserstein-based sketching objective that enforces full distributional shape, while retaining a variance term for scale control. By decoupling scale and shape, VISReg combines VICReg's flexibility with the distributional rigor of sketching methods, providing robust gradients even under collapse. We show that VISReg scales linearly, outperforms existing regularization on low-quality datasets, and is resilient to long-tailed and low-rank regimes. Pre-trained on ImageNet-1K, VISReg achieves state-of-the-art performance on out-of-distribution datasets. Pre-trained on ImageNet-22K, it matches DINOv2's OOD performance despite the latter using 10x more data (LVD-142M). Project and code: https://haiyuwu.github.io/visreg.

**Analysis:**

### 1. 摘要翻译
自监督学习（SSL）通过建模启发式方法或明确的正则化来防止嵌入崩溃。在后者中，VICReg将正则化分解为方差和协方差目标，提供了灵活性与可解释性。然而，协方差仅能捕获二阶统计量，虽然鼓励了解耦，但无法强制执行稳定的训练所需的完整分布形状。基于草图（Sketching）的方法（如SIGReg）通过将嵌入对齐到各向同性高斯分布解决了这一问题，但缺乏灵活性，且在嵌入崩溃时面临梯度消失。我们提出了方差-不变性-草图正则化（VISReg），它用基于切片Wasserstein距离（SWD）的草图目标取代了协方差，在保持方差项以进行尺度控制的同时，强制执行完整的分布形状。通过解耦尺度和形状，VISReg结合了VICReg的灵活性与草图方法的分布严谨性，即使在崩溃情况下也能提供稳健的梯度。VISReg在大规模数据集上线性缩放，在低质量数据集上表现出色，并对长尾和低秩区域具有鲁棒性。

### 2. 方法动机分析
*   **驱动力**：旨在消除自监督学习中对复杂启发式方法（如EMA、教师-学生架构）的依赖，同时解决现有正则化方法在分布约束上的不足。
*   **痛点**：
    *   **VICReg**：仅通过协方差正则化处理二阶统计量，无法保证嵌入空间的完整分布形状（各向同性），且计算复杂度为 $O(D^2)$。
    *   **SIGReg**：虽然利用Epps-Pulley检验实现分布对齐，但在嵌入空间崩溃时梯度会迅速减小直至消失，且无法有效解耦尺度与形状控制。
*   **研究假设**：通过引入基于最优传输的切片Wasserstein距离（SWD）对齐分布，并强制解耦“尺度（Scale）”与“形状（Shape）”的控制，可以获得既具备分布严谨性又具备稳健梯度的训练过程。

### 3. 方法设计详解
VISReg的正则化目标 $L_{Reg}$ 由尺度、形状和中心化三项组成：
*   **尺度正则化 (Scale Loss)**：计算各维度的标准差 $\sigma_j$，使其逼近1。通过中心化的嵌入 $Z$，公式为 $L_{scale} = \frac{1}{D} \sum_{j=1}^{D} (1 - \sigma_j(\hat{Z}))^2$。该项确保嵌入具有受控的缩放范围，且梯度不会消失。
*   **形状正则化 (Shape Loss)**：为了让分布形状趋向各向同性高斯，首先通过 $Z = \frac{\hat{Z}}{sg(\sigma) + \epsilon}$ 对尺度进行归一化（停止梯度），从而将几何结构与幅度解耦。利用SWD将嵌入投影到随机方向，计算其与标准高斯分布对应分位数之间的距离。
*   **中心化 (Center Loss)**：惩罚批均值的平方，增强训练的稳健性。
*   **流程总结**：
    1. 计算中心化项；
    2. 计算方差并执行尺度正则化；
    3. 通过停止梯度的标准差归一化嵌入；
    4. 对归一化嵌入进行随机投影并排序；
    5. 计算投影后的分位数误差作为形状损失。

### 4. 方法对比分析
*   **本质区别**：VISReg将非参数化的分布正则化引入训练，并使用最优传输理论（SWD）替代了VICReg的二阶协方差统计，通过解耦技术避免了梯度消失。
*   **创新点**：引入了基于SWD的形状正则化，并首次通过停止梯度（stop-gradient）实现了尺度与形状的彻底解耦。
*   **适用场景**：在大规模数据、低质量数据（如长尾分布、低秩数据）下表现优异，尤其适用于需要极高分布控制能力的自监督预训练任务。

### 5. 实验分析
*   **核心优势**：在OOD（分布外）泛化任务上达到SOTA；在ImageNet-22K上表现媲美DINOv2（仅需其1/10数据量）；在低质量数据集上具有极强的鲁棒性。
*   **局限性**：在部分in-domain任务上的线性探测精度略低于依赖复杂启发式的高性能基线（如iBOT）。

### 6. 实用指南
*   **开源信息**：已开源，代码参考：https://haiyuwu.github.io/visreg
*   **迁移建议**：该正则化模块可直接替换VICReg或类似Joint-Embedding架构中的正则化项。对于高噪声或小规模、长尾分布数据集，应适当调高 `lambda_shape` 的权重以增强形状约束。

### 7. 总结
*   **核心思想**：通过最优传输对齐分布，彻底解耦尺度与形状，实现稳健自监督学习。
*   **速记pipeline**：
    1. 计算嵌入均值并中心化；
    2. 计算方差并约束尺度至目标值；
    3. 归一化嵌入并将特征投影到随机维度；
    4. 对齐投影值与高斯分布的分位数，强制约束空间形状。

**Key Findings:**

- We propose Variance-Invariance-Sketching Regularization (VISReg), which replaces covariance with a Sliced-Wasserstein-based sketching objective that enforces full distributional shape, while retaining a variance term for scale control.
- We show that VISReg scales linearly, outperforms existing regularization on low-quality datasets, and is resilient to long-tailed and low-rank regimes.
- Pre-trained on ImageNet-1K, VISReg achieves state-of-the-art performance on out-of-distribution datasets.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02572v1)
- [arXiv](https://arxiv.org/abs/2606.02572v1)

---

<a id='2606.02551v1'></a>
## [AFUN: Towards an Affordance Foundation Model for Functionality Understanding](https://arxiv.org/abs/2606.02551v1)

**Authors:** Zhaoning Wang, Yi Zhong, Jiawei Fu, Henrik I. Christensen, Jun Gao

**Published:** 2026-06-01

**Categories:** cs.RO, cs.CV

**Abstract:**

Affordance understanding bridges visual perception and physical action, serving as an explainable interface for robot manipulation in open and unstructured real-world environments. Yet, building an affordance foundation model that not only understands where and how the interaction should happen, but also generalizes across diverse environments, objects, and tasks, remains a long-standing research challenge. Existing methods typically address only part of this challenge, either localizing task-relevant regions without specifying executable motion, or predicting motion but with limited scalability. In this paper, we present ourmodel, a step towards an affordance foundation model for functionality understanding. From a single RGB-D observation and a language task description, ourmodel predicts a task-conditional functional mask (where to interact) and a 3D post-contact motion curve (how to interact). To support open-world generalization, we build a large-scale standardized data pipeline that converts heterogeneous robot, human, simulation, and real-world scan data into a shared affordance schema with language, masks, and object-centric 3D motion labels. We evaluate ourmodel from three aspects: for affordance segmentation, ourmodel outperforms all baselines by a large margin across 8 test sets from 4 benchmarks, improving mean gIoU/cIoU by +23.9/+26.3; for contact-point prediction, it predicts substantially more accurate points, with a 12.7--61.3% hit-rate gain over the best baseline; and for 3D motion, it achieves the best performance on all three test sets. ourmodel can be deployed for real-world robot manipulation without finetuning for robot embodiment or using task-specific heuristics, demonstrating the ability to adapt to open-world affordance tasks. Project page: https://www.zhaoningwang.com/AFUN

**Analysis:**

作为计算机视觉和机器人领域的专家，我对 **AFUN (Affordance Foundation Model for Functionality Understanding)** 这篇论文的分析如下：

### 1. 核心贡献摘要
AFUN 提出了一种旨在实现“功能理解”的具身智能基础模型，能够根据单一 RGB-D 观测和语言指令，联合预测任务相关的**交互区域（功能掩码）**与**执行动作的 3D 运动轨迹**。该研究通过构建大规模、异构的标准化数据管线，实现了在不同环境、物体和任务间的泛化，为解决机器人操作在开放世界中的“如何交互”与“何处交互”提供了统一的接口。

### 2. 关键创新与方法论
*   **统一的功能表示范式：** 区别于传统方法将“感知（定位区域）”与“规划（生成轨迹）”割裂处理，AFUN 将其整合进单一模型架构中，直接输出“功能掩码 + 3D 轨迹”，实现了感知与决策的端到端对齐。
*   **异构数据标准化：** 核心创新在于其数据工程。该论文设计了一套能够将机器人操作数据、人类动作数据、仿真数据及 3D 扫描数据统一转化为“语言-掩码-3D 运动”标注的数据管线。这种跨来源的数据融合是模型实现强泛化能力的关键。
*   **零样本迁移能力：** 强调了无需针对特定机器人本体进行微调，也无需任务相关启发式规则（Heuristics），展现了强大的开放世界适应性。

### 3. 对领域的潜在影响
*   **从“任务执行”向“功能理解”的范式转移：** 此前的工作往往依赖于预定义的物体类别或特定的运动基元，而 AFUN 试图让模型理解物体的“功能（Affordance）”，这使得机器人能够处理未见过的工具和陌生的复杂环境。
*   **具身 AI 的数据范式：** AFUN 的数据管线构建思路为解决机器人领域“数据孤岛”问题提供了参考方案，即如何利用异构的非结构化数据训练通用策略。
*   **基准性能的突破：** 在多个 benchmarks 上实现超过 20% 的性能提升，证明了在大规模数据训练下的通用基础模型在具身智能领域的巨大潜力。

### 4. 受益的相关领域与应用
*   **服务机器人：** 在家庭或医院等非结构化环境下，机器人通过该模型可以更自然地与各类家居用品互动（如：理解不同杯子的把手位置及抓取动作）。
*   **工业自动化与柔性制造：** 减少了针对不同零部件进行繁琐的路径规划编码，通过语言指令即可快速切换不同任务。
*   **人机交互（HRI）：** 理解“功能”有助于机器人更好地预测人类意图，并辅助人类完成更复杂的协同任务。
*   **自动驾驶与无人机：** 在涉及与物理世界交互（如清理障碍物、操作开关等）的任务中，该方法具有极高的参考价值。

### 5. 潜在局限性（基于摘要分析）
*   **运动轨迹的动态适应性：** 虽然模型预测了 3D 运动曲线，但对于实时避障和执行过程中的闭环反馈控制（Closed-loop control）能力在摘要中并未明确说明。如果仅是开环预测，在复杂、动态场景下的鲁棒性可能受限。
*   **计算开销与实时性：** 作为一个大型基础模型，其在端侧部署时的推理延迟（Latency）可能对实时机器人控制提出挑战。
*   **交互序列的长程规划：** 摘要侧重于单一时刻的交互预测，对于需要多步骤协作才能完成的复杂任务（如：先开柜门再拿取物体），该模型是否具备长序列的逻辑推理能力仍需进一步观察。

**总结：** AFUN 的重要性在于其尝试将“语义感知”与“物理动作”通过一个统一的模型框架进行强绑定。如果该模型在实验中表现出的泛化能力确实能摆脱对特定机器人构型的依赖，那么它将是通往通用具身智能（General Purpose Embodied AI）的重要一步。

**Key Findings:**

- In this paper, we present ourmodel, a step towards an affordance foundation model for functionality understanding.
- We evaluate ourmodel from three aspects: for affordance segmentation, ourmodel outperforms all baselines by a large margin across 8 test sets from 4 benchmarks, improving mean gIoU/cIoU by +23.9/+26.3; for contact-point prediction, it predicts substantially more accurate points, with a 12.7--61.3% hit-rate gain over the best baseline; and for 3D motion, it achieves the best performance on all three test sets.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02551v1)
- [arXiv](https://arxiv.org/abs/2606.02551v1)

---

<a id='2606.02268v1'></a>
## [From Extrinsic to Intrinsic: Geodesic-Guided Representation Learning for 3D Geometric Data](https://arxiv.org/abs/2606.02268v1)

**Authors:** Yuming Zhao, Junhui Hou, Qijian Zhang, Jia Qin, Ying He

**Published:** 2026-06-01

**Categories:** cs.CV

**Abstract:**

Geometric analysis fundamentally distinguishes between \textit{extrinsic} and \textit{intrinsic} perspectives. The dominant paradigm in current 3D representation learning relies on either extrinsic spatial structures or high-level semantics, struggling to capture the essence of shape identity and underlying manifold topology. To bridge this gap, we introduce a novel 3D representation learning paradigm, namely \textbf{PRISM}, for \textbf{P}re-training, which learns isometric embeddings by \textbf{R}ecovering the \textbf{I}ntrinsic \textbf{S}urface geodesic \textbf{M}etric. PRISM incorporates a topology-enforcing objective that explicitly constrains the structure of latent space, alongside a specialized two-stage training recipe mitigating sample imbalance inherent in the distribution of geodesic distances. Experiments demonstrate that our approach shows satisfactory accuracy, robustness, and high efficiency in geodesic distance prediction and achieves superior performance across diverse downstream tasks, including shape recognition, surface parameterization, and non-rigid correspondence. The code will be publicly available at https://github.com/AidenZhao/PRISM.

**Analysis:**

### 1. 摘要翻译
几何分析在内蕴（intrinsic）与外蕴（extrinsic）视角之间存在根本区别。当前3D表示学习主要依赖外蕴空间结构或高层语义，难以捕捉形状本质及潜在的流形拓扑。为弥补这一差距，我们引入了一种名为 **PRISM** 的新颖预训练范式，旨在通过恢复内蕴表面测地线度量（Recovering the Intrinsic Surface geodesic Metric）来学习等距嵌入。PRISM 结合了强制 latent 空间拓扑结构的优化目标，以及一种旨在缓解测地线距离分布样本不平衡的专门两阶段训练方案。实验证明，该方法在测地线距离预测中表现出极高的精度、鲁棒性和效率，并在形状识别、曲面参数化和非刚性对应等下游任务中取得了优异性能。

### 2. 方法动机分析
*   **驱动力**：作者旨在构建一种能够捕获“形状不变性”的表示学习框架，使模型不仅能感知物体长什么样（外蕴），更能理解物体本身的几何拓扑（内蕴）。
*   **现有方法痛点**：现有主流方法多基于外蕴坐标或语义，缺乏对内蕴测地线度量的建模，导致在处理非刚性形变、 pose 变化等对拓扑结构敏感的任务时，性能受限。
*   **研究假设**：基于纳什嵌入定理（Nash Embedding Theorem），如果能将输入几何流形嵌入到一个保持测地线度量的特征空间中，那么该特征提取器将具备强大的几何感知能力。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **特征提取**：利用 Point Transformer V3 (PTv3) 将点云输入映射到高维特征空间 $\mathbb{R}^k$。
    2.  **测地线回归**：设计 MLP 解码器，通过特征差值 $\mathbf{h}_{ij} = |f_i - f_j|$ 预测点对间的测地线距离 $\hat{d}_{ij}$，并施加 L1 和相对误差约束。
    3.  **拓扑约束**：通过测地线结构一致性损失（$L_{struct}$），利用 `tanh` 近似符号函数，强制特征空间的距离序关系与原空间测地线距离序关系一致。
    4.  **两阶段训练**：先通过 $L_{struct}$ 进行全局拓扑结构预热，再通过基于测地线距离分布倒数加权的“重要性采样”进行微调。
*   **模型结构**：以 PTv3 为基础，通过测地线预测头进行自监督学习，模型本质上是在特征空间模拟一个“测地线场”。
*   **核心公式理解**：
    *   $L_{struct}$（公式6）：核心是利用 `sgn` 函数的软逼近，将“测地线距离序”编码进特征空间，确保拓扑相似的点在特征空间也更近。
    *   $w_{sample}$（公式9）：这是解决长尾分布的关键，对样本稀疏的距离区间（如极短或极远距离）给予更高权重，提升模型鲁棒性。

### 4. 方法对比分析
*   **本质区别**：不同于传统的对比学习（利用数据增强对齐）或 Masked Modeling（重建原始坐标），PRISM 引入了“测地线距离作为监督信号”，实现的是一种内蕴度量的等距嵌入。
*   **创新贡献**：首次实现了基于纯点云输入的高性能“预训练—微调”模式下的测地线预测，并成功推动了 Feed-forward（前馈式）表面参数化任务的发展。
*   **适用场景**：所有对几何拓扑敏感、涉及非刚性形变、曲面平滑或点对匹配的 3D 任务。

### 5. 实验分析
*   **验证方法**：在 ShapeNet（预训练）和 FAUST、ScanObjectNN（下游）数据集上验证。
*   **关键结果**：在测地线距离预测中，该方法在保持轻量化（Fast）的同时，精度逼近耗时极长的传统优化算法。在非刚性形变任务中，表现显著优于需要计算 Laplace-Beltrami 算子的方法。
*   **主要优势**：无需网格面片（Mesh）即可处理原始点云；推理速度极快（前馈式）；对噪声和非流形拓扑具有强鲁棒性。
*   **主要局限**：对极复杂的、存在大规模破洞或断裂的几何结构，内蕴度量恢复的稳定性仍有待提升。

### 6. 实用指南
*   **开源情况**：已开源（见论文链接）。
*   **实现细节**：重要性采样是提升效果的核心，预计算测地线分布并进行加权采样至关重要。需注意 `tanh` 参数 $\alpha$ 对损失函数收敛速率的影响。
*   **迁移可能**：该框架可以作为任何 3D 任务的通用预训练 Backbone，尤其是在需要精细几何理解的任务（如形变分析、骨骼提取）中具备极高的替代价值。

### 7. 总结
*   **核心思想**：通过测地线结构学习，将 3D 点云映射至保持内蕴拓扑的特征空间。
*   **速记版pipeline**：
    1.  提取特征。
    2.  回归测地线距离。
    3.  强制结构序一致性。
    4.  通过分布加权采样优化难点距离。

**Key Findings:**

- To bridge this gap, we introduce a novel 3D representation learning paradigm, namely \textbf{PRISM}, for \textbf{P}re-training, which learns isometric embeddings by \textbf{R}ecovering the \textbf{I}ntrinsic \textbf{S}urface geodesic \textbf{M}etric.
- Experiments demonstrate that our approach shows satisfactory accuracy, robustness, and high efficiency in geodesic distance prediction and achieves superior performance across diverse downstream tasks, including shape recognition, surface parameterization, and non-rigid correspondence.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02268v1)
- [arXiv](https://arxiv.org/abs/2606.02268v1)

---

<a id='2606.02577v1'></a>
## [RoboDream: Compositional World Models for Scalable Robot Data Synthesis](https://arxiv.org/abs/2606.02577v1)

**Authors:** Junjie Ye, Rong Xue, Basile Van Hoorick, Runhao Li, Harshitha Rajaprakash, Pavel Tokmakov, Muhammad Zubair Irshad, Vitor Guizilini, Yue Wang

**Published:** 2026-06-01

**Categories:** cs.RO, cs.CV

**Abstract:**

Scaling robot learning requires large-scale, diverse demonstrations, yet real-world data collection via teleoperation remains prohibitively expensive and time-consuming. While video diffusion models offer a promising avenue for data scaling, existing generative approaches are often limited to superficial visual augmentation, or suffer from embodiment hallucinations that yield physically infeasible motions. We present a generalizable embodiment-centric world model that achieves scalable data generation by synthesizing photorealistic demonstrations with novel objects, in novel scenes, and from novel viewpoints. Our approach anchors generation to rendered robot motion while conditioning on explicit scene and object priors, effectively decoupling trajectory execution from environment synthesis. This formulation has the potential to unlock two powerful data scaling capabilities: (1) retrieval and rebirth, which repurposes existing trajectories into entirely new contexts without new motion data; and (2) prop-free teleoperation, where operators manipulate empty air and the model hallucinates the target objects and scene afterwards, eliminating reset time. We demonstrate with real-world experiments that our generated data consistently improves downstream policy performance and significantly reduces real-world data requirements across diverse manipulation tasks.

**Analysis:**

### 1. 摘要翻译
机器人学习的规模化发展依赖于大规模、多样化的演示数据，但通过遥操作进行现实世界的数据采集既昂贵又耗时。虽然视频扩散模型为数据扩增提供了前景，但现有的生成方法往往局限于表层的视觉增强，或因缺乏对机器人本体（embodiment）的理解而产生物理上不可行的运动。我们提出了RoboDream，一种具有通用性的、以本体为中心的机器人世界模型。该模型通过将机器人运动的生成与环境上下文（即背景和对象）的生成解耦，实现了数据的可扩展合成。我们将生成过程锚定在渲染的机器人运动上，同时利用显式的场景和对象先验来驱动视觉合成。这种设计解锁了两种强大的数据扩展范式：(1)“检索与重生（Retrieval and Rebirth）”，它在不采集新运动数据的情况下，将现有轨迹复用到全新的场景中；(2)“无道具遥操作（Prop-free teleoperation）”，操作员在空旷环境中进行虚构操作，模型随后生成对应的对象和场景交互。实验表明，RoboDream能够持续提升下游策略性能，并显著降低真实世界的数据需求。

### 2. 方法动机分析
*   **驱动力**：旨在克服机器人学习中高昂的数据采集瓶颈，实现无需在目标任务上进行昂贵微调的零样本（zero-shot）数据生成。
*   **现有方法痛点**：现有工作要么受限于固定的轨迹（仅能修改视觉风格），要么由于无法解耦机器人运动与环境，导致在生成过程中产生“本体幻觉”（动作与物体交互不匹配、运动学不可行）。
*   **研究假设**：机器人操作任务中的动作、对象和场景是本质上独立且可重组的元素。通过显式解耦这三者并锚定运动学轨迹，可以在保持物理可行性的前提下，实现高度可控的视觉场景合成。

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入准备**：将机器人轨迹渲染为纯机器人的运动视频 $v_{rob}$（Kinematic Anchor），提取背景图像（Scene Prior $I_s$）和物体图像（Object Prior $I_o$）。
    2.  **多模态融合**：通过VAE编码器处理 $v_{rob}$ 和 $I_s$，将二者与视频隐空间 latent 拼接，作为基础输入；通过 MLP 处理全局轨迹状态 $\tau$。
    3.  **Cross-Attention注入**：利用任务描述 $l$ 和全局轨迹特征 $\tau$ 通过交叉注意力机制引导语义和运动的一致性。
    4.  **Self-Attention对象处理**：将对象先验 $I_o$ 编码为一组 latent tokens，注入到 Diffusion Transformer 的自注意力层中，使模型关注特定物体的细节。
    5.  **生成**：在上述条件的约束下，进行条件扩散生成。
*   **模型结构**：基于多模态视频扩散 Transformer，核心创新在于引入了“多模态通道扩展”（串联机器人运动和场景先验）和“对象先验注入”（通过注意力机制引入物体外观，实现不受空间位置约束的物体合成）。

### 4. 方法对比分析
*   **本质区别**：与传统视频补全或风格迁移不同，RoboDream 引入了“Kinematic Anchor”，即以渲染的运动学数据作为强制约束，确保生成的每一个动作在物理逻辑上是合理的。
*   **创新贡献**：解耦了“动（轨迹）”与“景（对象+背景）”，支持在零样本条件下完成环境重组和任务替换。
*   **适用场景**：适用于机器人操作任务的预训练数据增强，尤其在环境多样性受限或需要快速从已有轨迹迁移到新环境的场景。

### 5. 实验分析（精简版）
*   **关键结论**：在四个真实世界任务中，Gen-Mix（混合真实数据与RoboDream生成数据）的平均成功率从 36.3% 提升至 62.5%。
*   **主要优势**：实现了无道具（Prop-free）采集，不仅通过减少物理重置节约了约 50% 的时间，还通过零样本生成弥补了环境分布差异（Covariate Shift）。
*   **主要局限**：生成质量高度依赖于渲染的运动学轨迹准确性；对于极端复杂的交互动作（如绳索操纵），其基于帧的生成可能面临长程时序连贯性的挑战。

### 6. 实用指南
*   **开源情况**：项目主页为 https://junjieye.com/RoboDream/。
*   **实现建议**：
    *   **Prior Extraction**：训练时依赖 Grounded-SAM 进行自动目标识别和 OmniPaint 进行背景填充，这是构建训练数据集的关键步骤。
    *   **部署**：利用 Isaac Lab 进行轨迹的重新渲染是实现“视图泛化”的核心。
*   **迁移迁移**：核心思想可以迁移到任何需要“动作-场景”解耦的生成任务中，如自动驾驶中将轨迹复用到不同天气背景下。

### 7. 总结
*   **核心思想**：通过解耦运动约束与视觉上下文，实现可控的机器人数据合成。
*   **速记版Pipeline**：
    1. 提取物体和背景先验。
    2. 渲染基础机器人动作轨迹。
    3. 将轨迹、场景和物体注入模型。
    4. 通过扩散生成交互视频。

**Key Findings:**

- We present a generalizable embodiment-centric world model that achieves scalable data generation by synthesizing photorealistic demonstrations with novel objects, in novel scenes, and from novel viewpoints.
- Our approach anchors generation to rendered robot motion while conditioning on explicit scene and object priors, effectively decoupling trajectory execution from environment synthesis.
- This formulation has the potential to unlock two powerful data scaling capabilities: (1) retrieval and rebirth, which repurposes existing trajectories into entirely new contexts without new motion data; and (2) prop-free teleoperation, where operators manipulate empty air and the model hallucinates the target objects and scene afterwards, eliminating reset time.
- We demonstrate with real-world experiments that our generated data consistently improves downstream policy performance and significantly reduces real-world data requirements across diverse manipulation tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02577v1)
- [arXiv](https://arxiv.org/abs/2606.02577v1)

---

<a id='2606.02573v1'></a>
## [HumanNOVA: Photorealistic, Universal and Rapid 3D Human Avatar Modeling from a Single Image](https://arxiv.org/abs/2606.02573v1)

**Authors:** Hezhen Hu, Wangbo Zhao, Lanqing Guo, Hanwen Jiang, Jonathan C. Liu, Zhiwen Fan, Kai Wang, Zhangyang Wang, Georgios Pavlakos

**Published:** 2026-06-01

**Categories:** cs.CV

**Abstract:**

In this paper, we present HumanNOVA, a photorealistic, universal, and rapid model for generating 3D human avatars from a single RGB image. Achieving both photorealism and generalization is challenging due to the scarcity of diverse, high-quality 3D human data. To address this, we build a scalable data generation pipeline that follows two strategies. The first one is to leverage existing rigged assets and animate them with extensive poses from daily life. The second strategy is to utilize existing multi-camera captures of humans and employ fitting to generate more diverse views for training. These two strategies enable us to scale up to 100k assets, significantly enhancing both the quantity and the diversity of data for robust model training. In terms of the architecture, HumanNOVA adopts a feed-forward, token-conditioned avatar modeling framework that allows fast inference in less than one second and requires no test-time optimization. Given an input image and an estimated simplified human mesh (SMPL) without detailed geometry or appearance, the model first encodes both inputs into compact token representations. These tokens then act as conditioning signals and are fused through cross-attention to construct a triplane-based 3D avatar representation. Extensive experiments on multiple benchmarks demonstrate the superiority of our approach, both quantitatively and qualitatively, as well as its robustness under diverse input image conditions. Project page at https://HumanNOVA.github.io .

**Analysis:**

### 1. 摘要翻译
本文提出了 HumanNOVA，这是一个用于从单张 RGB 图像生成 3D 人体头像的写实、通用且快速的模型。实现写实感与泛化能力，由于缺乏高质量 3D 人体数据而极具挑战性。为此，我们构建了一个可扩展的数据生成流程，采用两种策略：一是利用现有的绑定资产结合日常动作进行动画化；二是利用现有的多视角人体捕捉数据进行拟合以产生多样化的训练视角。这两项策略使我们能够扩展至 10 万个资产，显著提升了数据规模和多样性。在架构方面，HumanNOVA 采用了前馈式的 Token 条件化建模框架，实现了不到一秒的快速推理，且无需测试时优化。模型通过将输入图像和粗略的人体网格（SMPL）编码为紧凑的 Token 表示，并通过交叉注意力机制融合，构建出基于三平面（Triplane）的 3D 人体表示。在多个基准测试上的实验验证了该方法的优越性及其在多种输入条件下的鲁棒性。

### 2. 方法动机分析
- **核心驱动力**：旨在解决单视图人体 3D 重建中“写实性”与“推理速度”难以兼得的问题。
- **现有方法痛点**：
    - **数据匮乏**：通用 3D 数据集规模大，但人体专用数据集规模较小（仅几千个实例），限制了大规模重建模型（LRM）的性能。
    - **推理延迟**：现有基于扩散模型或反复优化的方法，往往需要繁琐的测试时优化（Test-time Optimization）来细化几何与贴图，推理缓慢且泛化能力受限。
- **核心直觉**：通过极大规模、多样化的数据支撑，利用前馈式网络直接从图像中“推断”出高精度的 3D 表示，将耗时的优化过程内化为模型参数的学习。

### 3. 方法设计详解
- **流程总结**：
    1. **多模态编码**：分别使用 DINOv2 编码 RGB 图像得到视觉 Token，使用 PTv3 编码估计的 SMPL 网格得到网格 Token，引入人体先验。
    2. **2D-to-3D 映射**：将两组 Token 输入基于 Transformer 的映射网络，通过多层 Cross-Attention 融合，更新并重构出 3D 三平面（Triplane）表示（$T \in \mathbb{R}^{3hw \times d}$）。
    3. **渲染与训练**：利用标准的射线行进（Ray Marching）函数，从 triplane 渲染任意视角图像，并结合 RGB、Mask 和 LPIPS 损失函数进行端到端训练。
- **模型结构**：采用了 PointInfinity 的映射网络设计，通过多个 Building Block 进行特征融合与三平面细化。
- **关键先验**：利用 [16] 估计的 SMPL 网格提供粗略的结构引导，有效降低了单视角的歧义性。

### 4. 方法对比分析
- **本质区别**：与 SiTH 等需要测试时优化的方法不同，HumanNOVA 是一个**纯前馈（Feed-forward）模型**。
- **创新贡献**：
    - 构建了包含 10 万个资产的高质量合成+真实数据集。
    - 提出了将人体网格先验显式地集成进 Token 化流程的方法。
    - 在保证推理速度的同时，在视觉感知指标（LPIPS）上取得了显著提升。

### 5. 实验分析
- **验证方法**：在 CustomHuman、THuman2、2K2K 三大基准集上进行评估。
- **关键结论**：在保持 <1s 推理速度的前提下，相比最佳竞争对手 SiTH，LPIPS 指标提升超过 40%。
- **优势**：鲁棒性强，不仅能处理正面输入，在侧面输入及各种“野外”场景下表现稳定。
- **局限**：在极端复杂的遮挡情况或 SMPL 估计完全失效时，重建质量会下降。

### 6. 实用指南
- **开源/复现**：项目主页为 [HumanNOVA.github.io](https://HumanNOVA.github.io)。
- **实现建议**：
    - 训练需大规模算力（文中使用了 64 块 H100）。
    - 数据生成是关键：利用 AMASS 数据集进行动画化，利用 3DGS 进行多视角的拟合。
    -  Patch 采样策略对于显存优化至关重要，需通过前景比例计算采样权重。

### 7. 总结
- **核心思想**：海量数据支撑+网格先验引导的前馈式单图 3D 人体建模。
- **速记版 pipeline**：
    1. 获取图像与 SMPL 网格。
    2. 分别提取图像特征和网格特征。
    3. 通过 Transformer 将特征融合为三平面表示。
    4. 渲染目标视角图像并与地面真值计算损失。

**Key Findings:**

- In this paper, we present HumanNOVA, a photorealistic, universal, and rapid model for generating 3D human avatars from a single RGB image.
- Extensive experiments on multiple benchmarks demonstrate the superiority of our approach, both quantitatively and qualitatively, as well as its robustness under diverse input image conditions.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02573v1)
- [arXiv](https://arxiv.org/abs/2606.02573v1)

---

<a id='2606.02565v1'></a>
## [Policy-based Foveated Imaging and Perception](https://arxiv.org/abs/2606.02565v1)

**Authors:** Howard Xiao, Jan Ackermann, Boyang Deng, Gordon Wetzstein

**Published:** 2026-06-01

**Categories:** cs.CV

**Abstract:**

Ultra-high-resolution image sensors offer the potential to capture fine spatial details critical for many visual perception tasks, but acquiring and processing all pixels at full resolution is often infeasible under realistic bandwidth, latency, and power constraints. Existing approaches address this challenge through acquisition strategies such as spatial or temporal downsampling, which irrevocably discard information before task relevance can be assessed. In this work, we introduce a real-time, predictive, and task-aware foveated imaging system that operates directly at image acquisition time. Leveraging emerging dual-stream sensor architectures, our method dynamically allocates limited pixel bandwidth to task-relevant regions of interest while maintaining a low-resolution global context. We formulate foveated acquisition as a sensor attention policy-learning problem, in which past observations guide actions that determine future measurements, closing the perception-acquisition loop. Through extensive simulation across multiple perception tasks, we demonstrate that our approach achieves high task performance under strict pixel budgets and significantly outperforms relevant baselines operating at the same bandwidth. We further validate our system on a 200-megapixel dual-stream sensor, capturing real-world videos under realistic bandwidth and latency constraints, demonstrating the practical feasibility of task-driven, acquisition-time foveated imaging.

**Analysis:**

作为计算机视觉领域的专家，我对这篇题为《Policy-based Foveated Imaging and Perception》的论文进行了如下深度分析：

### 1. 论文核心贡献总结
该论文提出了一种**实时、任务驱动的中央凹（Foveated）成像系统**，旨在解决超高分辨率传感器在带宽、延迟和功耗限制下的数据处理难题。通过将传感器的采集过程建模为一个强化学习（RL）中的策略优化问题，该系统实现了在获取图像的同时动态分配带宽，从而在严苛的资源限制下，以最优方式保留任务相关区域的细节信息。

### 2. 关键创新与方法论
*   **闭环感知-采集系统（Perception-Acquisition Loop）：** 传统方法多为“先采集后处理”，而本文将“采集策略”本身作为感知系统的一部分。通过利用过去的信息来预测未来的测量需求，实现了采集端的闭环控制。
*   **基于策略的传感器注意力（Sensor Attention Policy）：** 作者将 foveated 采样建模为强化学习决策问题，系统学习如何通过动态分配“像素预算”来捕捉关键空间区域，这种“主动视觉”（Active Vision）的思想从像素获取阶段就开始执行，而非事后的图像处理。
*   **双流传感器架构（Dual-stream Sensor Architecture）：** 利用新型传感器硬件，同时保持低分辨率的全局上下文（Context）和高分辨率的兴趣区（ROI）捕捉，这种软硬结合的方式是实现该系统实时性的关键。

### 3. 对领域的潜在影响
这篇论文的意义在于它挑战了“全分辨率采集”的传统范式，为**资源受限环境下的计算机视觉**指明了新路径：
*   **范式转移：** 改变了以往“先通过下采样/压缩来丢弃信息”的做法，转而通过智能驱动硬件在采集源头进行“信息选择”。
*   **计算效能提升：** 在不牺牲感知性能的前提下，显著降低了后端 AI 处理流水线的带宽与计算负担，这对于自动驾驶、机器人视觉等对实时性和功耗极度敏感的领域具有里程碑意义。

### 4. 相关领域与应用前景
*   **自动驾驶与无人机：** 在处理超高分辨率视觉传感器时，实时过滤无关背景、聚焦目标（如障碍物、交通标志）将大幅提升系统响应速度。
*   **增强现实/虚拟现实（AR/VR）：** 该技术与人眼的注视点跟踪（Eye-tracking）结合，可以实现更高效的渲染与感知，极大地降低显示延迟。
*   **远程监控与安防：** 在带宽有限的网络环境下，实现对关键区域的高清监控，同时保障整体态势感知。
*   **边缘计算设备：** 使低功耗芯片处理超高清视觉数据成为可能，推动高精度感知向更小型的嵌入式终端普及。

### 5. 可推断的局限性
*   **硬件依赖性：** 论文高度依赖于“双流传感器架构”，这意味着该方法很难在现有的通用商业相机上通过软件直接复现，普及需要传感器的硬件迭代。
*   **复杂场景的泛化能力：** 当视野中同时存在大量瞬时变化的感兴趣目标时，策略模型可能会面临“带宽争抢”导致性能波动。
*   **训练与策略收敛：** 将复杂的视觉感知任务建模为强化学习策略学习，通常需要极其庞大且多样化的训练数据集，且针对不同任务（如从检测转向分割）是否需要重新训练策略也是一个潜在的挑战。

**专家总结：**
这篇论文的核心趣味性在于它模糊了“传感器”与“感知算法”之间的边界，将计算机视觉的前处理从静态的流水线变成了**动态的决策过程**。这种“按需采样”的哲学是未来高性能低功耗视觉感知系统的必然发展方向。

**Key Findings:**

- In this work, we introduce a real-time, predictive, and task-aware foveated imaging system that operates directly at image acquisition time.
- Leveraging emerging dual-stream sensor architectures, our method dynamically allocates limited pixel bandwidth to task-relevant regions of interest while maintaining a low-resolution global context.
- Through extensive simulation across multiple perception tasks, we demonstrate that our approach achieves high task performance under strict pixel budgets and significantly outperforms relevant baselines operating at the same bandwidth.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02565v1)
- [arXiv](https://arxiv.org/abs/2606.02565v1)

---

<a id='2606.02564v1'></a>
## [VLMs are Good Teachers for Video Reasoning via Adaptive Test-Time Optimization](https://arxiv.org/abs/2606.02564v1)

**Authors:** Junhao Cheng, Liang Hou, Tianxiong Zhong, Xin Tao, Pengfei Wan, Kun Gai, Jing Liao

**Published:** 2026-06-01

**Categories:** cs.CV

**Abstract:**

The recent "Reasoning with Video" paradigm utilizes Video Generation Models (VGMs) to generate temporally coherent visual trajectories to complete reasoning tasks. Although state-of-the-art VGMs excel at visual quality, they often struggle to understand and follow task-specific rules, leading to logical failures across diverse reasoning scenarios. Existing efforts try to utilize Vision-Language Models (VLMs) as problem pre-solvers to produce or refine textual guidance for the VGM. However, textual descriptions fail to capture intricate spatiotemporal details, and VGMs often struggle to faithfully execute fine-grained or long-tail instructions even with a valid plan. While VLMs struggle as solvers, they possess strong perception capabilities to evaluate process-constraint satisfaction and final-goal achievement. Leveraging this strength, we introduce a paradigm shift that transitions the role of VLMs to "teachers". Specifically, a VLM teacher extracts task-specific rules to formulate differentiable rewards, guiding a VGM Reasoner via test-time online optimization of a lightweight LoRA module. This strategy enables adaptive test-time optimization and extends the reasoning capabilities beyond the VGM's intrinsic boundaries. Evaluations on symbolic (VBVR-Bench) and general-purpose (RULER-Bench) video reasoning benchmarks show that the proposed method yields a 16.7-point average performance gain, outperforming the VLM-as-Solver paradigm (+0.4 points) and Best-of-N scaling (+2.2 points) by a large margin at comparable test-time cost. These findings reveal that integrating VLMs as test-time teachers offers a promising paradigm for achieving generalizable video reasoning. Project Page: https://VLM-as-Teacher.github.io/

**Analysis:**

### 1. 摘要翻译
“Reasoning with Video”范式利用视频生成模型（VGMs）生成时间上连贯的视觉轨迹来完成推理任务。尽管现代VGM在视觉质量上表现优异，但它们在理解和遵循特定任务规则方面往往表现不佳。现有方法通常使用视觉语言模型（VLMs）作为解题者（Solver）来辅助，但纯文本指导难以捕捉复杂的时空约束，且VGM难以将高层指令转化为细粒度的视觉执行。本文提出了一个范式转移，将VLM的角色转变为“教师（Teacher）”。VLM教师从任务描述中提取规则，制定可微奖励，并通过在线测试时优化（TTO）指导VGM推理器。在VBVR-Bench和RULER-Bench上的实验表明，该方法平均性能提升了16.7个点，大幅优于现有的VLM-as-Solver和Best-of-N策略，且测试时计算成本可控。

### 2. 方法动机分析
*   **驱动力**：旨在解决视频生成模型在进行复杂、规则受限的推理任务时（如逻辑一致性、物理约束遵循）的表现不足问题。
*   **现有方法痛点**：
    *   **VLM作为解题者（Solver）**：基于文本的提示往往无法精确描述复杂的时空约束，且VGM难以忠实执行细粒度指令（即文本与执行之间的鸿沟）。
    *   **测试时缩放（如Best-of-N）**：仅通过增加采样次数缓解随机性，无法修正模型本质的逻辑推理缺陷或系统性偏差。
*   **研究假设**：VLM虽然不擅长直接生成执行轨迹，但具备极强的感知能力来验证过程约束和最终目标达成情况，因此更适合担任“监督者”而非“解题者”。

### 3. 方法设计详解
*   **流程总结**：
    1.  **监督合成（Task-Adaptive Supervision）**：VLM接收任务输入，提取目标和过程规则，并将其转换为以“Yes”为目标的二元奖励查询（Reward Queries）。
    2.  **在线优化（Online Optimization）**：利用冻结的VLM教师作为评价器，通过可微VQA损失直接计算梯度，更新轻量级LoRA模块。
    3.  **推理生成**：优化后的LoRA引导VGM生成最终视频。
*   **核心技术细节**：
    *   **可微奖励函数**：将视频-查询对的VLM预测损失作为损失函数，直接反向传播到LoRA参数。
    *   **高效设计**：
        *   **轻量化解码**：在线评估时使用 surrogate decoder，减少内存和显存开销。
        *   **蒸馏推理**：将VGM蒸馏为四步生成器，优化仅针对第一步的“纯噪声到干净潜在空间”预测，利用早期推理特征，大幅缩减计算量。
        *   **早停机制**：基于VLM的综合置信度（即loss小于阈值）提前终止优化。

### 4. 方法对比分析
*   **本质区别**：从传统的“提示词工程（Prompting）”或“采样搜索（Sampling）”转向“基于模型反馈的参数级测试时优化（TTO）”。
*   **创新贡献**：首次引入了VLM-as-Teacher paradigm，证明了通过反向传播VLM的评估反馈来在线适配生成模型的可行性与卓越表现。
*   **适用场景**：规则严苛、涉及复杂时空交互或物理状态变化的视觉推理任务。

### 5. 实验分析
*   **验证方法**：在VBVR-Bench（符号推理）和RULER-Bench（通用任务）上进行大规模评估。
*   **关键结论**：在保持可比计算成本下，平均性能提升达16.7%，证明了该方法在规则遵循和逻辑一致性上的显著优越性。
*   **优势**：实现了对复杂规则的细粒度视觉对齐；克服了文本描述的歧义性。
*   **局限**：强依赖VLM的感知准确度；若VLM未察觉某些错误，优化信号将无效。

### 6. 实用指南
*   **开源/实现**：项目主页为 https://VLM-as-Teacher.github.io/。实现关键在于如何构建高质量的过程约束查询。
*   **关键超参**：$\lambda$（目标与过程约束的平衡，默认0.5）、LoRA Rank（16）、学习率（$5 \times 10^{-5}$）。
*   **迁移建议**：可迁移至任何需要精准控制生成结果、符合特定规则的任务中，如辅助设计、特定风格视频生成等。

### 7. 总结
*   **核心思想**：通过VLM反向传播的可微奖励，在推理时在线优化生成模型。
*   **速记版Pipeline**：
    1.  提取规则转化为奖励查询；
    2.  生成初步视频片段；
    3.  VLM评价视频是否合规；
    4.  计算损失并微调LoRA；
    5.  利用更新后的模型生成最终结果。

**Key Findings:**

- Although state-of-the-art VGMs excel at visual quality, they often struggle to understand and follow task-specific rules, leading to logical failures across diverse reasoning scenarios.
- Leveraging this strength, we introduce a paradigm shift that transitions the role of VLMs to "teachers".

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02564v1)
- [arXiv](https://arxiv.org/abs/2606.02564v1)

---

<a id='2606.02553v1'></a>
## [LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation](https://arxiv.org/abs/2606.02553v1)

**Authors:** Qixin Hu, Shuai Yang, Wei Huang, Song Han, Yukang Chen

**Published:** 2026-06-01

**Categories:** cs.CV

**Abstract:**

Autoregressive (AR) video diffusion enables variable-length synthesis, but long-horizon generation often suffers from accumulated errors and identity drift. For efficiency, existing methods commonly adopt sliding-window attention during generation. This creates an irreversible generation trajectory: once the active window accumulates appearance errors, subsequent generations can only condition on this degraded trajectory and drift further away. We address this limitation by formulating long video generation as a retrieval-augmented generation (RAG) problem. Rather than relying solely on the recent window, we treat previously generated latents as a dynamic, searchable history. We propose LongLive-RAG, a general retrieval framework for AR video generation. At each new block, LongLive-RAG uses a query embedding to retrieve relevant historical latents. This lightweight retrieval step adds only a small overhead relative to generation and lets the generator condition on non-local context instead of only the recent window. To make retrieval more discriminative, we introduce the Window Temporal Delta Loss that suppresses redundant local similarity and encourages embeddings to capture meaningful temporal changes. Together, these components help reduce error accumulation caused by sliding-window attention. Experiments across multiple AR backbones and generation lengths show improved long-video quality and the best average VBench-Long rank. To our knowledge, among open-ended AR long video generation methods, LongLive-RAG is the first to formulate self-generated latent history as content-addressable retrieval memory. Code is available at https://github.com/qixinhu11/LongLive-RAG.

**Analysis:**

# 论文分析：LongLive-RAG

## 1. 摘要翻译
自回归（AR）视频扩散模型能够实现变长合成，但长序列生成常受累积误差和身份漂移的困扰。为保证效率，现有方法通常采用滑动窗口注意力机制。这导致了不可逆的生成轨迹：一旦活动窗口积累了外观错误，后续生成只能基于该降级轨迹，进而导致更严重的漂移。我们通过将长视频生成建模为检索增强生成（RAG）问题来解决这一限制。我们提出的LongLive-RAG框架将已生成的潜空间向量（latents）视为可动态搜索的历史记录。在每个新块生成时，LongLive-RAG使用查询嵌入检索相关的历史潜空间信息，并将其作为额外的非局部上下文。为了使检索更具判别性，我们引入了窗口时间增量损失（Window Temporal Delta Loss）来抑制冗余的局部相似性，并鼓励嵌入捕获有意义的时间变化。在多个AR主干网络和生成长度上的实验表明，LongLive-RAG显著提升了长视频质量，并在VBench-Long指标上获得了最佳平均排名。

## 2. 方法动机分析
*   **驱动力**：解决长视频生成中“滑动窗口”机制导致的累积误差和身份一致性丧失问题，通过利用长程历史信息而非仅仅依靠最近的窗口来辅助当前生成。
*   **现有方法痛点**：滑动窗口遗弃了早期的生成历史，一旦近期窗口内出现外观畸变或身份漂移，生成器只能在错误基础上继续，形成不可逆的恶性循环。
*   **研究假设**：如果生成器能够在生成过程中“回头看”已生成的、高质量的历史片段（而非仅依赖压缩后的状态），就能有效纠正当前漂移并维持长程一致性。

## 3. 方法设计详解
*   **流程总结**：
    1.  **离线训练检索器**：训练一个冻结生成器基础上的轻量级潜空间编码器（Autoencoder），将高维潜空间向量映射为1024维检索嵌入。
    2.  **构建检索池**：在生成过程中，每完成一个数据块，将其潜空间向量通过编码器生成嵌入并存入搜索池（Key为向量，Value为对应的原始上下文）。
    3.  **动态检索（Query）**：在生成新块前，使用最近一个块的嵌入作为Query，计算与池中历史嵌入的余弦相似度，检索Top-K相关块。
    4.  **上下文拼接**：将检索到的历史上下文（Mt）与滑动窗口上下文（Cloc）及可选固定Sink（Csink）拼接，构成完整的注意力上下文序列，送入原生成器。
*   **核心算法**：
    *   **Window Temporal Delta Loss**：解决邻近帧过于相似导致检索浪费的问题。通过拉开短时间窗内帧的嵌入距离，强迫检索器关注更有判别性的长程信息。
    *   **Smoothness Penalty**：平滑嵌入轨迹，防止检索源在时间轴上跳变过大。

## 4. 方法对比分析
*   **本质区别**：传统方法侧重于“历史如何压缩”（如压缩Token、滑动窗口），而LongLive-RAG侧重于“历史如何查找”，将视频历史从“流”变成了“数据库”。
*   **创新贡献**：首次将生成的潜空间历史作为内容可寻址（content-addressable）的检索内存，且无需微调原生成器模型，具有极强的通用性。
*   **适用场景**：所有基于自回归（AR）架构的视频扩散模型，特别是要求长序列一致性的视频生成任务。

## 5. 实验分析
*   **验证方法**：在Causal-Forcing、Self-Forcing和LongLive三种主流AR主干上，对比∞-RoPE、Deep Forcing等基线，并在30s、60s、120s长度上评估。
*   **关键结果**：在VBench-Long所有任务中排名第一，显著改善了主体和背景的一致性。
*   **局限**：由于是检索增强，其质量上限依然受限于基础AR模型的能力；且检索池会随视频时长增长而增大，虽然通过压缩嵌入缓解了存储压力，但检索耗时随视频长度增加。

## 6. 实用指南
*   **开源情况**：代码已开源至 GitHub (qixinhu11/LongLive-RAG)。
*   **实现细节**：建议使用预训练的编码器并配合 `LSeqDelta` 损失训练，K=6是平衡上下文内容与流畅度的最佳超参数。
*   **迁移可能**：该框架与特定生成模型解耦，理论上可轻松迁移至任何利用潜空间进行自回归预测的视频生成任务中。

## 7. 总结
*   **核心思想**：通过检索已生成的潜空间历史来辅助自回归视频生成，实现长程一致性校正。
*   **速记版pipeline**：
    1. 编码已生成的视频块为特征嵌入；
    2. 存入历史库；
    3. 生成时检索最相似的历史片段；
    4. 拼接到当前注意力机制中引导生成。

**Key Findings:**

- We propose LongLive-RAG, a general retrieval framework for AR video generation.
- At each new block, LongLive-RAG uses a query embedding to retrieve relevant historical latents.
- To make retrieval more discriminative, we introduce the Window Temporal Delta Loss that suppresses redundant local similarity and encourages embeddings to capture meaningful temporal changes.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02553v1)
- [arXiv](https://arxiv.org/abs/2606.02553v1)

---

<a id='2606.02535v1'></a>
## [LL-Bench: Rethinking Low-Level Vision Evaluation in the Era of Large-Scale Generative Models](https://arxiv.org/abs/2606.02535v1)

**Authors:** Lu Liu, Huiyu Duan, Chenxin Zhu, Jintong Lu, Haoyun Jiang, Liu Yang, Qiang Hu, Guangtao Zhai, Xiaoyun Zhang

**Published:** 2026-06-01

**Categories:** cs.CV

**Abstract:**

Large-scale generative models have demonstrated remarkable capabilities across image generation and editing tasks. However, their performance in low-level vision tasks, which require pixel-wise control, remains insufficiently studied. To address this gap, we introduce \textbf{LL-Bench}, a comprehensive \textbf{Benchmark} for evaluating the capabilities of large-scale generative models on \textbf{L}ow-\textbf{L}evel vision tasks. The benchmark comprises 2,469 real-world degraded images covering 16 low-level degradation tasks, and 28,919 restored images produced by 10 state-of-the-art large-scale generative models and 21 conventional restoration models, which are annotated with 152,020 expert-level pairwise human preferences and 28,334 quality scores. Built upon LL-Bench, we present a systematic diagnosis that reveals the performance boundaries and unique failure modes of large-scale generative models across diverse low-level vision tasks, compared with conventional representative restoration approaches. Moreover, we investigate the effectiveness of current quality evaluation metrics on LL-Bench, which exhibit significant discrepancy with human ratings. To better align restored-image quality assessment with human preferences, we further propose \textbf{LL-Score}, an MLLM-based evaluator that captures both restoration quality and hallucination existence. Extensive experiments demonstrate that LL-score not only outperforms existing image quality assessment metrics, but also serves as a promising reward model for training generative models on low-level vision tasks.

**Analysis:**

### 1. 摘要翻译
大规模生成模型在图像生成和编辑任务中表现卓越，但其在需要像素级控制的低层视觉任务中的表现尚缺乏充分研究。为填补这一空白，我们提出了**LL-Bench**，这是一个用于全面评估大规模生成模型在低层视觉任务中能力的基准。该基准包含2,469张真实世界退化图像，涵盖16个低层退化任务，以及由10个顶尖大规模生成模型和21个传统复原模型生成的28,919张复原图像，并附带152,020个专家级成对人类偏好标注和28,334个质量评分。基于LL-Bench，我们进行系统诊断，揭示了大规模生成模型与传统代表性复原方法相比的性能边界和独特失败模式。此外，我们研究了现有质量评估指标在LL-Bench上的表现，发现其与人类评价存在显著偏差。为更好地将复原图像质量评估与人类偏好对齐，我们进一步提出了**LL-Score**，这是一个基于MLLM的评估器，能捕捉复原质量和幻觉存在。实验证明，LL-Score不仅优于现有图像质量评估指标，还可作为训练生成模型进行低层视觉任务的有效奖励模型。

### 2. 方法动机分析
- **驱动力**：利用大规模生成模型（LGM）强大的先验知识解决低层视觉任务，同时解决生成式模型在“视觉真实感”与“内容保真度”之间的矛盾。
- **现有方法痛点**：传统的全参考（FR）和无参考（NR）指标严重依赖确定性重建或仅关注感知质量，无法识别生成模型常见的“幻觉”现象，且与人类主观偏好脱节。
- **研究假设**：通过融合人类偏好数据训练多模态语言模型（MLLM），能够构建一个能够同时评估“复原质量”与“幻觉存在”的通用评估器（LL-Score），并将其作为强化学习的奖励模型。

### 3. 方法设计详解
**LL-Score 架构：**
- **特征提取**：利用 Qwen3-VL 的视觉编码器（冻结）提取退化输入 $I_s$ 和复原输出 $I_r$ 的特征，并通过可训练的投影层（Projector）映射至语言空间。
- **协同推理**：在文本提示词（Prompt）后加入两个特殊Token `<|QUAL|>` 和 `<|HAL|>`，分别对应质量评估和幻觉检测。
- **双头预测**：由两个轻量级 MLP 头基于这些隐藏状态分别输出质量得分 $s_q$ 和幻觉概率 $p_h$。

**算法细节：**
- **rank-aware joint fine-tuning（联合微调策略）**：
  - **质量头**：采用带边际的 Bradley-Terry 损失函数，不仅考虑成对偏好，还通过人类排名差异动态调整边际（margin），确保高质量图像得分显著高于低质量图像。
  - **幻觉头**：使用二元焦点损失（Binary Focal Loss）处理类别不平衡问题，强制模型输出幻觉概率。

### 4. 方法对比分析
- **本质区别**：从传统的数学指标（PSNR/SSIM）转向基于专家人类偏好的MLLM判别模型。
- **创新贡献**：提出首个针对低层视觉的幻觉检测与质量评估统一框架，并证明其作为奖励模型能有效引导生成模型进行“忠实度”优化。
- **适用场景**：适用于各种图像复原场景（去噪、超分、去雨、去雾等）的模型评估与在线策略优化。

### 5. 实验分析
- **核心结论**：LL-Score 在 SRCC（相关性）和 Acc（分类准确率）上全面超越传统指标和现有MLLM基线。
- **主要优势**：极强的通用性，通过一个模型即可处理16种不同视觉任务，且对幻觉的识别率远超现有零样本大模型。
- **主要局限**：作为MLLM模型，评估推理速度较传统算术指标慢，且性能受限于基础模型（Qwen3-VL）的推理能力。

### 6. 实用指南
- **开源情况**：代码与数据集已在 GitHub 公开（`https://github.com/MediaX-SJTU/LL-Bench`）。
- **实现建议**：复现时重点在于构建成对偏好数据，训练过程中使用 LoRA 进行高效微调，并务必通过 attention check 机制剔除不合格的标注者以保证数据质量。
- **迁移性**：LL-Score 的架构可直接迁移至医疗图像质量评估或专业领域图像复原评估，只需更换 Prompt 和微调数据。

### 7. 总结
- **核心思想**：利用MLLM对齐人类偏好，统一评估图像复原质量与幻觉。
- **速记版pipeline**：
  1. 提取退化/复原图像视觉特征。
  2. 结合任务提示词与特殊Token输入大模型。
  3. 双MLP头分别输出质量得分与幻觉概率。
  4. 利用排名损失与焦点损失进行联合训练。

**Key Findings:**

- To address this gap, we introduce \textbf{LL-Bench}, a comprehensive \textbf{Benchmark} for evaluating the capabilities of large-scale generative models on \textbf{L}ow-\textbf{L}evel vision tasks.
- The benchmark comprises 2,469 real-world degraded images covering 16 low-level degradation tasks, and 28,919 restored images produced by 10 state-of-the-art large-scale generative models and 21 conventional restoration models, which are annotated with 152,020 expert-level pairwise human preferences and 28,334 quality scores.
- Built upon LL-Bench, we present a systematic diagnosis that reveals the performance boundaries and unique failure modes of large-scale generative models across diverse low-level vision tasks, compared with conventional representative restoration approaches.
- Extensive experiments demonstrate that LL-score not only outperforms existing image quality assessment metrics, but also serves as a promising reward model for training generative models on low-level vision tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02535v1)
- [arXiv](https://arxiv.org/abs/2606.02535v1)

---

<a id='2606.02510v1'></a>
## [Not All Points Are Equal: Uncertainty-Aware 4D LiDAR Scene Synthesis](https://arxiv.org/abs/2606.02510v1)

**Authors:** Xiang Xu, Alan Liang, Youquan Liu, Xian Sun, Linfeng Li, Lingdong Kong, Ziwei Liu, Qingshan Liu

**Published:** 2026-06-01

**Categories:** cs.CV, cs.RO

**Abstract:**

Constructing faithful 4D worlds from LiDAR-acquired sequences is crucial for embodied AI, yet current generative frameworks apply uniform modeling capacity across all spatial regions. This ignores that perceptual difficulty varies dramatically within a single scan: distant surfaces, occluded boundaries, and small-scale objects carry far higher uncertainty than well-observed structures. We present U4D, a new framework that explicitly leverages spatial uncertainty to guide LiDAR scene generation in a "hard-to-easy" schedule. U4D derives per-point uncertainty maps via Shannon Entropy from a pretrained segmentor, then applies an unconditional diffusion stage to synthesize high-entropy areas with precise geometry, followed by a conditional completion stage that fills in the remaining regions using these structures as priors. A MoST (Mixture of Spatio-Temporal) block further maintains cross-frame coherence by dynamically balancing spatial detail and temporal continuity. Extensive experiments on nuScenes and SemanticKITTI demonstrate state-of-the-art scene fidelity, temporal consistency, and downstream performance.

**Analysis:**

### 1. 摘要翻译
从激光雷达（LiDAR）序列构建高保真4D场景对具身智能至关重要，但目前的生成框架在所有空间区域均施加统一的建模能力。这忽略了感知难度在单次扫描内的巨大差异：远距离表面、被遮挡边界和小型目标比可见结构具有更高的不确定性。我们提出了U4D，这是一种新框架，通过显式利用空间不确定性来指导“由难到易”的LiDAR场景生成策略。U4D利用预训练分割器的香农熵导出逐点不确定性图，通过无条件扩散阶段以精确几何结构合成高熵区域，随后通过条件补全阶段以这些结构为先验填补剩余区域。MoST（时空混合）块通过动态平衡空间细节与时间连续性，进一步保持跨帧的一致性。在nuScenes和SemanticKITTI上的广泛实验证明了该方法在场景保真度、时间一致性和下游感知性能方面的先进性。

### 2. 方法动机分析
*   **驱动力**：作者认为“并非所有点都同等重要”，应将计算资源倾斜于高难度的感知区域。
*   **现有方法痛点**：现有LiDAR生成模型将空间视为均匀分布，导致在生成远距离物体、被遮挡区域时产生“幽灵”现象、物体闪烁或语义混乱。
*   **研究假设**：通过显式识别场景中的感知难点（高熵区域）并优先建模这些区域，可以作为“结构锚点”，从而显著提升整体生成质量与一致性。

### 3. 方法设计详解
*   **流程总结**：
    1.  **不确定性测量**：利用预训练分割器获取逐点分类概率，通过香农熵（Shannon Entropy）生成不确定性图，选取Top-K点构建稀疏不确定区域表示 $x_0^u$。
    2.  **阶段1：无条件生成（Hard）**：使用扩散模型根据不确定性图生成高熵区域的精确几何结构。训练采用带有二进制掩码监督的去噪损失。
    3.  **阶段2：条件补全（Easy）**：将Stage 1生成的结构作为先验，通过条件扩散模型填补剩余空间。输入为带噪声的完整场景和Stage 1的输出。
    4.  **MoST融合**：在扩散骨干网络中引入MoST模块，动态融合空间特征（空间卷积）和时间特征（时间卷积）。
*   **模型结构**：核心在于双阶段扩散架构与MoST gating机制。MoST通过MLP预测门控权重（$\alpha^s, \alpha^t$），实现对空间与时间信息的自适应加权。
*   **算法解释**：MoST gating的设计借鉴了混合专家系统（MoE），其物理意义在于：在网络底层和输出层，空间细节（几何）是主要的；在中间层，时间动态（运动一致性）起主导作用。

### 4. 方法对比分析
*   **本质区别**：从传统的“均匀生成”范式转变为“以不确定性为先导的层级生成”范式。
*   **创新贡献**：首次将不确定性度量引入4D LiDAR生成流程，提出“硬到易”的生成调度，有效解决了长距离生成不稳定的难题。
*   **适用场景**：自动驾驶、机器人导航等需要高精度、时序一致的LiDAR环境模拟任务。

### 5. 实验分析
*   **关键结论**：在nuScenes数据集上，FRD指标较基线提升6%–11%，且在长时序上表现出最优的TTCE指标（时间一致性）。
*   **主要优势**：极大减少了长距离物体的伪影，提升了下游感知任务的准确率。
*   **主要局限**：高度依赖预训练分割器的性能；双阶段生成增加了推理开销。

### 6. 实用指南
*   **开源情况**：论文提到该框架在nuScenes/SemanticKITTI验证，代码逻辑在文中阐述明确，建议关注SenseTime相关开源库（如OpenDWM）。
*   **实现细节**：$\lambda \mathcal{L}_{\mathrm{mask}}$ 是平衡几何细节与整体结构的关键；MoST块中的正则化项（系数变动正则化）对于防止门控退化至关重要。
*   **迁移可能**：该思想可直接迁移至RGB-D视频生成或雷达感知任务，核心在于将“不确定性指导”作为Attention mask或条件输入。

### 7. 总结
*   **核心思想**：基于不确定性驱动“由难到易”的层级化4D生成。
*   **速记版pipeline**：
    1. 计算区域不确定性。
    2. 优先生成复杂结构。
    3. 补全剩余简单背景。
    4. 时空动态门控融合。

**Key Findings:**

- We present U4D, a new framework that explicitly leverages spatial uncertainty to guide LiDAR scene generation in a "hard-to-easy" schedule.
- Extensive experiments on nuScenes and SemanticKITTI demonstrate state-of-the-art scene fidelity, temporal consistency, and downstream performance.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.02510v1)
- [arXiv](https://arxiv.org/abs/2606.02510v1)

---

