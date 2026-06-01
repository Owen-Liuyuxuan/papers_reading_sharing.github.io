time: 20260601

# Arxiv Computer Vision Papers - 2026-06-01

## Executive Summary

### 执行摘要：2026-05-29 Arxiv 计算机视觉论文精选

#### 一、主要主题与趋势
本期10篇论文覆盖了计算机视觉与机器人学的多个前沿方向，整体呈现三大趋势：

1. **视频理解与生成中的时序与频率建模**：多篇论文聚焦如何更高效地处理视频数据中的时序一致性（如 #1 双手动作分割、#2 视频统一模型的频域桥接、#6 视频对象中心学习的时序内化），表明从“空间感知”向“时空联合建模”的深入演进。
2. **具身智能与机器人操控**：机器人领域论文激增（#7 机载规划、#9 地形感知全身控制、#10 表面约束策略），结合视觉感知与物理约束，推动从“仿真环境”到“真实世界部署”的跨越。
3. **视觉基座模型的评估与推理能力拓展**：基准测试与数据集创新活跃（#3 语义对应关系基准、#4 长尾自动驾驶推理），强调对视觉模型“理解能力”而非单纯“识别能力”的评估。

#### 二、显著创新论文
- **#4 nuReasoning**：首个针对自动驾驶长尾场景的推理中心数据集与基准，将“因果推理”引入视觉安全评估，可能推动自动驾驶从感知级向认知级跃迁。
- **#1 Polyphony**：提出交替视觉Transformer与语义条件实现双手动作分割，在细粒度时序理解任务中引入多模态条件控制，方法设计新颖。
- **#5 RayDer**：从真实世界视频中实现可扩展的自监督新视角合成，无需显式3D监督，对神经渲染向实际应用落地具有重要意义。

#### 三、新兴研究方向与技术
- **频率域桥梁**：如 #2 Lumos-Nexus 利用同构潜空间在频域桥接视频模型，预示“频域建模”可能成为统一视频理解的通用范式。
- **推理与因果性**：除 #4 外，多篇机器人论文（如 #9、#10）开始显式建模环境约束与因果关系，而非仅依赖端到端学习。
- **轻量化与实时性**：#7 机载规划消除推理冗余，#8 FSM-Net 在频域-空间域融合实现高效去模糊，体现对边缘设备部署的持续关注。
- **无显式正则化的时序内化**：#6 探索免额外约束下视频对象中心学习的时序一致性，为无监督时序学习提供新思路。

#### 四、推荐精读论文（按优先级排序）
1. **#4 nuReasoning** —— 对自动驾驶安全研究者和推理模型设计者最具启发。
2. **#5 RayDer** —— 对神经渲染、新视角合成方向的研究者必读。
3. **#1 Polyphony** —— 对交互式动作分析、双手建模领域有高参考价值。
4. **#3 SOCO** —— 对评估视觉基座模型的能力边界至关重要，适合所有使用预训练模型的研究者。
5. **#9、#10** —— 对四足机器人操控与环境交互方向，建议合并阅读以对比不同约束策略。

总体而言，本期论文体现了CV领域从“感知性能竞赛”向“推理、因果与真实世界部署”的转向，机器人学与视觉的交叉研究正在加速成熟。

---

## Table of Contents

1. [Polyphony: Diffusion-based Dual-Hand Action Segmentation with Alternating Vision Transformer and Semantic Conditioning](#2605.31115v1)
2. [Lumos-Nexus: Efficient Frequency Bridging with Homogeneous Latent Space for Video Unified Models](#2605.31603v1)
3. [SOCO: Benchmarking Semantic Object Correspondence in Vision Foundation Models](#2605.31597v1)
4. [nuReasoning: A Reasoning-Centric Dataset and Benchmark for Long-Tail Autonomous Driving](#2605.31572v1)
5. [RayDer: Scalable Self-Supervised Novel View Synthesis from Real-World Video](#2605.31535v1)
6. [Internalizing Temporal Consistency in Video Object-Centric Learning without Explicit Regularization](#2605.31508v1)
7. [On-Device Robotic Planning: Eliminating Inference Redundancy for Efficient Decision-Making](#2605.31460v1)
8. [FSM-Net: An Efficient Frequency-Spatial Network for Real-World Deblurring](#2605.31400v1)
9. [Learning Terrain-Aware Whole-Body Control for Perceptive Legged Loco-Manipulation](#2605.31343v1)
10. [Surface Constraint Policy for Learning Surface-Constrained and Dynamically Feasible Robot Skills](#2605.31321v1)

---

## Papers

<a id='2605.31115v1'></a>
## [Polyphony: Diffusion-based Dual-Hand Action Segmentation with Alternating Vision Transformer and Semantic Conditioning](https://arxiv.org/abs/2605.31115v1)

**Authors:** Hao Zheng, Hu Wang, Tiantian Zheng, Prajjwal Bhattarai, Tuka Alhanai

**Published:** 2026-05-29

**Categories:** cs.CV

**Abstract:**

Dual-hand action segmentation, densely predicting actions for both hands from untrimmed videos, is essential for understanding complex bimanual activities. However, it poses several unique challenges: complex inter-hand dependencies, visual asymmetry between hands, representation conflicts where the dominant hand monopolizes gradients, and semantic ambiguity in fine-grained actions. We propose Polyphony, a three-stage method to address these challenges through: (1) an Alternating Dual-Hand Vision Transformer that alternates training between left- and right-hand mini-batches to ensure balanced gradient contributions from both hands while sharing a spatio-temporal encoder; (2) Semantic Feature Conditioning that aligns visual features with structured, compositional action descriptions to enhance discrimination of semantically similar actions; and (3) Diffusion-Based Segmentation with cross-hand feature fusion for inter-hand coordination and adaptive loss weighting for balancing performance. Polyphony achieves state-of-the-art on both dual-hand datasets (HA-ViD, ATTACH) with improvements up to 16.8 points, and on the single-stream Breakfast dataset (82.5%), outperforming the prior best method that uses a 12x larger backbone. Notably, our unified model with a single shared backbone surpasses baselines requiring separate per-hand models. Code is at https://github.com/x-labs-xyz/Polyphony-Dual-hand-Action-Segmentation.

**Analysis:**

### 1. 摘要翻译
双手动作分割（即从无剪辑视频中为左右手密集预测动作）对于理解复杂的双手活动至关重要。然而，该任务存在多重独特挑战：复杂的双手依赖关系、双手间的视觉不对称性、主导手导致的梯度更新冲突以及细粒度动作的语义模糊性。为此，我们提出了Polyphony。该方法包含三部分：(1) **交替式双手视觉Transformer (ADH-ViT)**，通过在左右手批次间交替训练，在共享编码器的同时确保梯度贡献平衡；(2) **语义特征调节**，通过对齐结构化动作描述来增强对相似动作的辨别能力；(3) **基于扩散的分割模型**，通过跨手特征融合实现交互协调，并通过自适应损失权重平衡性能。Polyphony在HA-ViD和ATTACH数据集上达到SOTA，并在Breakfast数据集上以显著更小的模型超越了现有最佳方案。

---

### 2. 方法动机分析
*   **驱动力**：人类操作通常是双手协同的，但现有动作分割模型大多仅针对单手或单序列，忽视了双手间复杂的时空依赖与协调动态。
*   **痛点**：1) 联合建模时，主导手（通常为右手）易垄断梯度，导致非主导手训练不足；2) 动作语义细微差异（如“拧螺母到螺栓” vs “拧螺母到轴”）仅靠视觉难以区分；3) 现有模型依赖昂贵的人工标注（如包围盒）。
*   **核心直觉**：将双手动作视作一组互补的旋律（Polyphony），在共享感知基座的同时，通过交替训练机制打破梯度偏见，并利用语言语义空间纠正视觉歧义。

---

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **ADH-ViT特征提取**：利用共享的VideoMAE V2作为骨干，通过管状嵌入（Tubelet Embedding）处理视频块。
    2.  **语义调节（Semantic Conditioning）**：引入结构化的语言描述（动词+操作对象+目标+工具），通过TCN对齐模块将语义嵌入注入视觉特征。
    3.  **扩散分割（Diffusion-based Segmentation）**：将动作标签预测转化为条件扩散生成过程。模型首先融合跨手特征，随后通过去噪解码器逐步细化动作分布。
*   **核心算法**：
    *   **交替训练策略**：在第$j$步，基于 $\lfloor j/\Delta \rfloor \pmod 2$ 选择 LH 或 RH 数据集进行梯度更新，而非同时采样。这有效防止了“单手独大”。
    *   **语义对齐损失**：结合余弦相似度（方向对齐）与MSE（幅度对齐），确保语义描述空间与视觉特征空间的强一致性。
    *   **自适应权重**：根据验证集近期表现，通过Boost因子实时调整左右手损失权重，动态修复性能滑坡。

---

### 4. 方法对比分析
*   **本质区别**：不同于以往将双手视为独立任务或需要额外框标注的方案，Polyphony是端到端的统一建模，强调跨手交互与动态梯度平衡。
*   **创新贡献**：提出ADH-ViT训练策略解决梯度冲突，将语义结构化知识融入动作分割的扩散模型中。
*   **适用场景**：适用于复杂的协作装配、手术机器人操作等涉及双手细粒度协作的视频分析任务。

---

### 5. 实验分析
*   **关键结论**：在HA-ViD上，Polyphony将左右手准确率分别提升了12.0和16.8个点；在Breakfast数据集上，以86M参数的ViT-Base优于1B+参数的EAST模型。
*   **主要优势**：极佳的平衡能力，模型统一性强，语义引导显著提升了细粒度辨别力。
*   **局限性**：仍存在对复杂协调行为的过度同步预测偏见，且语义描述依赖手动解析，自动化程度有待提升。

---

### 6. 实用指南
*   **开源与实现**：代码已开源（GitHub链接见原文）。实现时需注意 `lclip` 滑动窗口采样和 `∆` 交替周期的调优。
*   **迁移建议**：若迁移至多智能体协作任务，可将原双手的“左右手分支”替换为“多代理分支”，并增加图神经网络模块增强代理间的通信。
*   **训练细节**：必须采用预训练的VideoMAE权重作为起点；语义 conditioning 阶段训练需先于扩散阶段。

---

### 7. 总结
*   **核心思想**：通过语义引导与交替梯度平衡实现双手协同感知的统一扩散模型。
*   **速记版 Pipeline**：
    1. 共享骨干提取双手视觉特征；
    2. 引入结构化语言知识纠正语义模糊；
    3. 左右手训练轮流切换防偏倚；
    4. 扩散过程融合双手信息生成动作标签。

**Key Findings:**

- We propose Polyphony, a three-stage method to address these challenges through: (1) an Alternating Dual-Hand Vision Transformer that alternates training between left- and right-hand mini-batches to ensure balanced gradient contributions from both hands while sharing a spatio-temporal encoder; (2) Semantic Feature Conditioning that aligns visual features with structured, compositional action descriptions to enhance discrimination of semantically similar actions; and (3) Diffusion-Based Segmentation with cross-hand feature fusion for inter-hand coordination and adaptive loss weighting for balancing performance.
- Polyphony achieves state-of-the-art on both dual-hand datasets (HA-ViD, ATTACH) with improvements up to 16.8 points, and on the single-stream Breakfast dataset (82.5%), outperforming the prior best method that uses a 12x larger backbone.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31115v1)
- [arXiv](https://arxiv.org/abs/2605.31115v1)

---

<a id='2605.31603v1'></a>
## [Lumos-Nexus: Efficient Frequency Bridging with Homogeneous Latent Space for Video Unified Models](https://arxiv.org/abs/2605.31603v1)

**Authors:** Jiazheng Xing, Hangjie Yuan, Lingling Cai, Xinyu Liu, Yujie Wei, Fei Du, Hai Ci, Tao Feng, Jiasheng Tang, Weihua Chen, Fan Wang, Yong Liu

**Published:** 2026-05-29

**Categories:** cs.CV, cs.AI

**Abstract:**

Connector-based video unified models have demonstrated strong capability in instruction-grounded video synthesis, but integrating a large high-fidelity generator into the unified training loop is computationally prohibitive, limiting achievable visual quality. We therefore propose Lumos-Nexus, a training-efficient unified video generation framework that facilitates the development of strong reasoning-driven generation capabilities while significantly enhancing visual fidelity. Lumos-Nexus adopts a two-stage design: 1) During training, only a lightweight generator is aligned with the understanding block to learn to take in reasoning-driven semantic control. 2) During inference, we introduce Unified Progressive Frequency Bridging (UPFB) to progressively hand off generation to a high-capacity pretrained generator in the shared latent space, enabling coarse-to-fine refinement and producing high-fidelity videos without compromising reasoning quality. To fill the gap in reasoning-driven video generation benchmarks, we introduce VR-Bench, which assesses a model's capability to translate inferred intent into coherent and semantically aligned video content. Extensive experiments demonstrate that Lumos-Nexus achieves substantial gains in visual realism and temporal coherence on VBench, while exhibiting strong reasoning-based generative performance on VR-Bench. Code and models are available at https://jiazheng-xing.github.io/nexus-lumos-home/.

**Analysis:**

作为计算机视觉领域的专家，我针对《Lumos-Nexus》这篇论文的分析如下：

### 1. 主要贡献总结
Lumos-Nexus 提出了一种兼顾“高阶逻辑推理”与“高质量视觉生成”的统一视频生成框架。其核心贡献在于通过两阶段设计，将轻量级生成器的语义对齐能力与高性能生成器的高保真度结合，解决了在大规模统一模型训练中因计算资源限制而导致的视觉质量瓶颈。

### 2. 关键创新点与方法论
*   **训练效率优化（两阶段设计）：** 训练阶段仅对轻量级生成器进行对齐，使其聚焦于学习推理驱动的语义控制，极大降低了算力开销。
*   **统一渐进频率桥接（UPFB）：** 这是本文最亮眼的技术创新。在推理阶段，利用“同构潜在空间（Homogeneous Latent Space）”，通过频率域的渐进式传递，将语义生成任务从轻量模型无缝切换至高容量预训练模型，实现“由粗到细”的视频精炼。
*   **评估体系突破（VR-Bench）：** 针对现有视频生成评估多关注视觉外观而非逻辑连贯性的痛点，作者引入了 VR-Bench，专门衡量模型对复杂意图的理解及语义对齐能力。

### 3. 对领域的潜在影响
*   **打破“推理”与“质量”的跷跷板效应：** 该研究为视频生成模型提供了一条路径，使模型既能听懂复杂的指令（如“根据一段对话逻辑生成后续情节”），又能输出影院级的画面。
*   **资源普惠化：** 证明了无需全量微调昂贵的大型生成器即可获得高保真视频，这对于学术界和中小型实验室具有重要的工程借鉴意义。
*   **推动多模态模型进化：** 这种“桥接式”架构可能成为未来通用视频大模型（Universal Video Models）的标准范式，即解耦“语义决策”与“视觉渲染”。

### 4. 相关领域与应用价值
*   **长视频内容创作：** 极高的逻辑连贯性使得该框架非常适合自动电影制作、动漫番剧自动生成等复杂剧情任务。
*   **智能交互与辅助系统：** 在机器人视频反馈、自动驾驶场景仿真中，模型需要精准响应逻辑指令，该方法提供的语义控制能力将大有用武之地。
*   **虚拟现实（VR/AR）：** 对时空一致性的要求较高，UPFB 带来的高质量视觉输出能显著提升虚拟环境的真实感。

### 5. 可推断的局限性
*   **频率转换的精度挑战：** “频率桥接（Frequency Bridging）”依赖于两个模型在潜在空间中具有极高的同构性，若模型间的特征分布存在差异，可能会导致转换过程中出现伪影（Artifacts）或高频细节丢失。
*   **推理延迟：** 虽然训练是高效的，但推理阶段采用了“两段式生成”（先轻量后重型），这意味着在追求高保真度时，推理延迟（Latency）可能比单模型架构更高，难以满足实时交互需求。
*   **对预训练模型的依赖：** UPFB 的表现高度依赖于所接入的“高容量预训练生成器”本身的性能上限，若该生成器自身缺乏泛化能力，则整体架构的上限将受限。

**专家点评：**
这篇论文的有趣之处在于它敏锐地捕捉到了当前视频生成领域的痛点：**“脑子灵光但手笨（推理强但画质差）”与“手艺精湛但脑子死板（画质强但逻辑差）”之间的矛盾**。通过频率空间的巧妙过渡，Lumos-Nexus 实际上是在尝试一种“跨模型协作”的机制，这在生成式 AI 领域是一个非常有价值的探索方向。

**Key Findings:**

- 2) During inference, we introduce Unified Progressive Frequency Bridging (UPFB) to progressively hand off generation to a high-capacity pretrained generator in the shared latent space, enabling coarse-to-fine refinement and producing high-fidelity videos without compromising reasoning quality.
- To fill the gap in reasoning-driven video generation benchmarks, we introduce VR-Bench, which assesses a model's capability to translate inferred intent into coherent and semantically aligned video content.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31603v1)
- [arXiv](https://arxiv.org/abs/2605.31603v1)

---

<a id='2605.31597v1'></a>
## [SOCO: Benchmarking Semantic Object Correspondence in Vision Foundation Models](https://arxiv.org/abs/2605.31597v1)

**Authors:** Olaf Dünkel, Basavaraj Sunagad, Haoran Wang, David T. Hoffmann, Christian Theobalt, Adam Kortylewski

**Published:** 2026-05-29

**Categories:** cs.CV

**Abstract:**

Measuring structured object understanding in vision foundation models remains challenging due to inconsistent evaluation protocols and limited part-level supervision. Semantic correspondence (SC) evaluates this capability by testing whether object parts can be matched across instances and categories under large variations in appearance, viewpoint, and geometry. To enable a systematic SC evaluation, we introduce SOCO, a new benchmark for Semantic Object Correspondence that introduces a taxonomy of correspondence types and provides consistent, functionally meaningful keypoint annotations across 100 categories and over 1M correspondence pairs. In addition, SOCO includes keypoint language descriptions, enabling the evaluation of large vision-language models (LVLMs) and their fine-grained part-level understanding. Comprehensive experiments reveal that (i) vision foundation backbones encode strong semantic structure but transfer correspondences poorly across related categories and only partially capture object-part position, (ii) LVLMs are stronger at text-prompted part localization than at visual-reference cross-image matching, exposing a gap between language-grounded localization and fine-grained visual correspondence, and (iii) correspondence performance predicts performance on dense downstream tasks, including segmentation, tracking, 3D pose estimation, and 3D detection, more strongly than ImageNet classification. Together, these findings position SOCO as a benchmark for structured, part-level representation quality in vision and multimodal foundation models.

**Analysis:**

以下是关于《SOCO: Benchmarking Semantic Object Correspondence in Vision Foundation Models》的论文分析。

---

### 1. 摘要翻译
测量视觉基础模型中结构化对象理解的能力，因评估协议不一致和缺乏部分级标注而颇具挑战。语义对应（SC）评估通过测试对象部件在外观、视点和几何发生巨大变化时，能否在不同实例和类别之间匹配来衡量该能力。为实现系统的SC评估，我们引入了SOCO，一个新的语义对象对应基准。SOCO引入了一种对应关系类型的分类法，在100个类别和超过100万个对应对中提供一致、功能上有意义的键点标注。此外，SOCO还包含键点语言描述，支持评估大视觉语言模型（LVLMs）及其细粒度部分级理解能力。实验表明：(i) 视觉基础骨干模型编码了强语义结构，但跨类别的对应迁移较差，且仅部分捕获了对象部件位置；(ii) LVLMs 在文本提示的部分定位方面比视觉参考的跨图像匹配更强，揭示了语言接地定位与细粒度视觉对应之间的差距；(iii) 对应性能比ImageNet分类更强地预测了密集的下游任务（分割、跟踪、3D位姿估计和3D检测）。

### 2. 方法动机分析
*   **驱动力**：现有的语义对应（SC）评估不仅缺乏跨类别的对应逻辑，且将“识别局部概念（如轮子）”与“识别特定部件（如左前轮）”的能力混为一谈，无法量化模型到底在哪个环节失效。
*   **痛点**：现有数据集（如SPair-71k）的标注多为几何启发式定义而非语义定义，缺乏可复用的层级分类法，导致无法进行跨类别评估。
*   **研究假设**：结构化对象理解能力可以通过将语义概念与几何位置显式解耦，并利用语言描述将其与跨模态模型对齐来准确探测。

### 3. 方法设计详解
*   **SOCO 核心分类法**：作者提出将语义对应分解为三个维度：
    1.  **概念对应 (CC)**：识别相同的局部概念（如“轮子”）。
    2.  **语义对象对应 (SOC)**：识别同一概念且具有相同对象相对身份（如“左前轮”）。
    3.  **跨类别 SOC (Cross-SOC)**：利用分类法跨越不同类别（如将汽车的轮子与卡车的轮子对应）。
*   **流程细节**：
    1.  **数据构建**：基于ImageNet，对100个类别进行人工标注，采用UI辅助保证一致性。
    2.  **语言描述**：通过元组`(category, concept, position_in_part, position_in_object)`自动生成自然语言描述（如“公交车左前轮的中心点”）。
    3.  **评估协议**：
        *   **视觉模型**：通过零样本最近邻特征匹配计算PCK（Percentage of Correct Keypoints）。
        *   **LVLM**：构建多项选择VQA任务，通过CircularEval协议（四种答案排列组合）严格评估对语言描述与视觉对应的一致性。

### 4. 方法对比分析
*   **本质区别**：与传统SC数据集不同，SOCO是一个“分类法驱动”而非“几何定义驱动”的基准。
*   **创新点**：首次在SC评估中引入语言描述，实现了对视觉基础模型（VFMs）与多模态模型（LVLMs）的统一评估框架。
*   **场景**：极佳的零样本表征质量诊断工具，尤其适用于评估模型在机器人操纵（如工具使用）和细粒度理解方面的能力。

### 5. 实验分析
*   **结论**：
    1.  **视觉模型失效模式**：强视觉骨干模型（如DINOv2）能识别概念，但在处理“重复部件混淆（CC→SOC差距）”时显著下降。
    2.  **LVLM失效模式**：LVLMs更擅长处理文字提示的定位（Desc.设置），而在视觉参考（Vis.设置）下表现较差，存在明显的视觉与语言对齐缝隙。
    3.  **预测能力**：SOC性能比ImageNet kNN更能准确预测密集任务（分割、跟踪、3D检测）的表现。

### 6. 实用指南
*   **开源**：代码与数据集已开源，访问 [https://genintel.github.io/SOCO/](https://genintel.github.io/SOCO/)。
*   **注意事项**：在进行模型对比时，应关注CC到SOC的性能跌幅，这比单纯看平均分数更能揭示模型对几何结构的编码质量。
*   **迁移**：该基准可以作为任何视觉Encoder训练的“检查站”，通过快速零样本评估确定模型是否产生了真正的结构化表征。

### 7. 总结
*   **核心思想**：通过解耦语义与几何属性，以分类法重塑语义对应评估。
*   **速记版pipeline**：
    1. 标注语义概念及位置层级。
    2. 生成对应的自然语言描述。
    3. 分别计算概念匹配与实例匹配分数。
    4. 对比视觉与语言模态下的差异性能。

**Key Findings:**

- To enable a systematic SC evaluation, we introduce SOCO, a new benchmark for Semantic Object Correspondence that introduces a taxonomy of correspondence types and provides consistent, functionally meaningful keypoint annotations across 100 categories and over 1M correspondence pairs.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31597v1)
- [arXiv](https://arxiv.org/abs/2605.31597v1)

---

<a id='2605.31572v1'></a>
## [nuReasoning: A Reasoning-Centric Dataset and Benchmark for Long-Tail Autonomous Driving](https://arxiv.org/abs/2605.31572v1)

**Authors:** Zhiyu Huang, Johnson Liu, Rui Song, Zewei Zhou, Ruining Yang, Yun Zhang, Tianhui Cai, Hanyin Zhang, Mingxuan Gao, Valeria Xu, Jiali Chen, Yishan Shen, Yiluan Guo,  Tony,  Qi, Jiaqi Ma

**Published:** 2026-05-29

**Categories:** cs.CV

**Abstract:**

Reasoning is essential for autonomous driving (AD) in long-tail scenarios, where vehicles must apply commonsense knowledge, understand spatial relations, infer agent interactions, and make safe decisions. However, existing AD datasets and benchmarks mainly target perception, prediction, or planning, and provide limited supervision for reasoning over realistic long-tail driving scenes. We introduce nuReasoning, a large-scale real-world dataset and benchmark for reasoning-centric AD. Following the lineage of nuScenes and nuPlan, nuReasoning advances real-world AD datasets and benchmarks toward reasoning in long-tail driving scenarios. The dataset contains 20,000 clips, each 20 seconds long, collected across multiple cities, with synchronized multi-camera images, LiDAR data, HD maps, object annotations, and human-verified reasoning annotations spanning Spatial Reasoning, Decision Reasoning, and Counterfactual Reasoning. Unlike prior datasets that focus primarily on visual question answering, nuReasoning supports both reasoning evaluation and planning evaluation, enabling a direct study of how reasoning supervision affects driving performance. Experiments show that fine-tuning VLMs on nuReasoning substantially improves driving-specific question answering, while incorporating reasoning supervision into VLA training improves planning performance even when textual reasoning outputs are disabled at inference time. These results establish nuReasoning as a foundation for evaluating and improving robust, interpretable, reasoning-driven AD systems in realistic long-tail settings.

**Analysis:**

### 1. 摘要翻译
推理在长尾自动驾驶场景中至关重要，要求车辆应用常识、理解空间关系、推断交互并做出安全决策。然而，现有的自动驾驶数据集主要针对感知、预测或规划，缺乏针对长尾场景的系统性推理监督。我们推出了 **nuReasoning**，这是一个用于推理中心化自动驾驶的大规模数据集和基准。该数据集包含20,000个20秒片段，涵盖多城市、多模态传感器数据，并提供人机协同验证的空间推理、决策推理和反事实推理标注。实验表明，在nuReasoning上微调视觉语言模型（VLM）显著提升了驾驶专用问答能力；将推理监督融入视觉-语言-动作（VLA）模型训练，即使在推理时关闭文本推理输出，也能显著提高规划性能。

### 2. 方法动机分析
*   **驱动力**：旨在弥合当前端到端驾驶模型仅依赖轨迹模仿训练导致的“长尾场景泛化能力弱、缺乏可解释性”的鸿沟。
*   **现有痛点**：现有数据集（如nuScenes, nuPlan）主要关注感知与轨迹规划，虽有少量QA数据集，但缺乏与下游规划任务直接关联的因果、反事实推理监督。
*   **核心假设**：引入结构化推理标注（空间、决策、反事实）作为中间任务监督，能引导模型学习到更鲁棒的场景表示，从而间接优化规划表现。

### 3. 方法设计详解
*   **数据 mining pipeline**：
    1.  **自动挖掘**：利用Gemini 3.1 Pro对Fleet logs进行评分（1-10分），筛选高难度长尾场景。
    2.  **人工验证**：专家确认长尾属性并选定关键帧（keyframe），确保数据质量。
*   **标注维度**：
    *   **空间推理**：识别物体、3D状态、语义关系（距离、方向）。
    *   **决策推理**：解释车辆行为逻辑，输出Meta-action（纵向+横向）及因果追溯（Reasoning Trace）。
    *   **反事实推理**：枚举安全/不安全替代动作，分析潜在风险（"What-if"分析）。
*   **nuVLA模型结构**：
    *   **VLM Backbone**：Qwen3-VL-2B-Instruct，用于编码多模态输入，输出推理文本。
    *   **Trajectory DiT**：基于流匹配（Flow-Matching）的决策Transformer头，通过Cross-attention融合VLM特征，输出5秒规划轨迹。
    *   **训练策略**：多任务联合优化，推理监督（Reasoning loss）与动作规划（Action loss）协同。

### 4. 方法对比分析
*   **本质区别**：不仅关注视觉问答，更强调**推理与规划的强耦合**，通过推理任务监督作为“辅助监督”提升规划模型的表征能力。
*   **创新点**：首次系统性引入“反事实评估”标注，强制模型在驾驶决策中评估替代方案的安全性，这对长尾环境的避险至关重要。
*   **场景适用**：特别适用于复杂城市、施工路段、异常交通流等需要高阶常识理解的场景。

### 5. 实验分析（精简版）
*   **验证方法**：在nuReasoning测试集上评估VLM的 reasoning accuracy 以及 nuVLA 的 planning score (NPS)。
*   **关键结论**：
    1.  推理监督能显著提升VLA的规划指标（NPS从64.98提升至73.09），且这种增益在推理时禁用文本输出依然存在，证明了模型内部学到了更好的表示。
    2.  空间推理与决策推理组合能带来最稳健的提升。
*   **优劣势**：优势在于泛化能力强，解释性高；局限在于对极细粒度空间坐标的回归仍有一定难度（需进一步优化）。

### 6. 实用指南
*   **开源**：项目官网：[https://nureasoning.github.io/](https://nureasoning.github.io/)。
*   **迁移建议**：可将nuReasoning作为辅助预训练集，通过LoRA适配器将推理监督引入现有的VLA架构中，作为“强化学习RLHF”之前的有效中间监督阶段。

### 7. 总结
*   **核心思想**：利用多维结构化推理数据监督提升自动驾驶模型的场景理解与避险规划能力。
*   **速记版pipeline**：
    1.  利用大模型筛选长尾片段；
    2.  人工标注空间、决策与反事实数据；
    3.  结合推理监督训练VLM与流匹配规划器；
    4.  通过安全门控与规划指标评估模型性能。

**Key Findings:**

- We introduce nuReasoning, a large-scale real-world dataset and benchmark for reasoning-centric AD.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31572v1)
- [arXiv](https://arxiv.org/abs/2605.31572v1)

---

<a id='2605.31535v1'></a>
## [RayDer: Scalable Self-Supervised Novel View Synthesis from Real-World Video](https://arxiv.org/abs/2605.31535v1)

**Authors:** Ulrich Prestel, Stefan Andreas Baumann, Nick Stracke, Björn Ommer

**Published:** 2026-05-29

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Self-supervised novel view synthesis (NVS) remains challenging to scale, despite the abundance of video data, largely due to the brittleness of training on realistic videos and the hard-to-predict scaling behavior of multi-network system designs. We introduce RayDer, a unified, feed-forward transformer that consolidates camera estimation, scene reconstruction, and rendering into a single backbone, turning self-supervised NVS into a well-posed single-model scaling problem. A minimal dynamic state, treated as a nuisance factor, absorbs time-varying content and enables stable training on unconstrained real-world video. Importantly, RayDer keeps static-scene NVS as its target task: dynamic content is leveraged purely as scalable supervision, not reconstructed as in dynamic-scene (4D) NVS. Across multiple model sizes and orders of magnitude in data, RayDer exhibits clean power-law scaling with data and compute, and outperforms static-scene data mixtures. On a large number of benchmarks, RayDer achieves strong zero-shot open-set performance competitive with state-of-the-art supervised approaches. Project Page: https://compvis.github.io/rayder

**Analysis:**

# RayDer: 基于海量动态视频的可扩展自监督新视角合成

## 1. 摘要翻译
自监督新视角合成（NVS）虽得益于海量视频数据，但受限于在真实视频上训练的脆弱性以及多网络系统难以预测的缩放行为。本文引入了 RayDer，一个统一的、前馈 Transformer 架构，将相机估计、场景重建和渲染整合进单一骨干网，将自监督 NVS 转化为一个良好定义的单模型缩放问题。通过将动态状态建模为一种“干扰因素（nuisance factor）”，系统能吸收时间变化内容，实现对无约束现实世界视频的稳定训练，同时保持静态场景 NVS 为目标任务。RayDer 在不同模型规模和数据量下表现出优异的幂律缩放规律，在多个基准测试上实现了与监督学习方法相当的零样本开放集性能。

## 2. 方法动机分析
- **驱动力**：解决现有自监督 NVS 方法因依赖稀缺的静态场景数据集而无法真正“缩放”的问题，通过设计一种能够处理动态视频的架构，解锁大规模互联网视频数据的潜力。
- **现有方法痛点**：
    1. **多网络管道复杂性**：现有方法通常由多个相互交互的独立网络组成，难以优化且缩放行为不可预测。
    2. **对静态场景的病态依赖**：直接在包含动态物体的真实视频上训练会导致模型性能崩坏或无法收敛。
- **研究假设**：通过架构统一化（Single-network）并显式处理动态内容（将其作为“干扰因素”而非重建对象），可以将 NVS 转化为一个标准的高性能深度学习缩放问题。

## 3. 方法设计详解
### 流程总结
1. **统一化架构**：将原先用于相机估计、场景重建和渲染的三个独立 ViT 整合进一个统一的 Transformer 主干网。
2. **干扰因素建模**：引入一个轻量级的动态状态嵌入 $s_i$，随同相机姿态预测一同产生。$s_i$ 在训练时吸收动态物体的残留特征，防止其干扰相机姿态的表示，推理时移除该变量。
3. **自回归位姿学习**：为了防止模型在多视图训练时利用帧序列的时间顺序作为捷径，采用随机顺序的自回归方式预测位姿，迫使模型学习真实的几何关系。
4. **并行目标注意力机制**：针对每个目标视图单独处理会带来巨大计算消耗，通过特殊的注意力遮罩（Mask），使输入视图相互感知，而目标视图仅关注输入视图，从而实现高效的 KV 缓存与并行预测。

### 模型结构与算法
- **统一模型 $\mathcal{M}$**：输入为无标注视频帧，输出为位姿 $\{p_i\}$ 和状态 $\{s_i\}$。所有计算均由共享主干完成，通过自适应归一化（Adaptive Norms）区分任务。
- **动态状态丢失函数**：在训练中随机以零向量替换 $s_i$（Dropout），迫使模型即使在缺乏状态输入时也能渲染出合理的静态视图。

## 4. 方法对比分析
- **本质区别**：不试图重建动态场景（非 4D NVS），而是通过“动态状态抑制”技术，在动态数据中提纯出稳定的静态几何表示。
- **创新贡献**：
    1. **架构 unification**：完全摒弃了传统的“估计-重建-渲染”多网络 pipeline。
    2. **动态干扰建模**：提出了处理大规模动态视频数据的简单而鲁棒的机制。
    3. **Compute-optimal Scaling**：首次在自监督 NVS 领域确立了模型参数、数据量与计算量之间的幂律缩放关系。

## 5. 实验分析
- **关键结果**：RayDer 在 RE10K 等数据集上展现出明显的缩放性能，数据量增加（1% -> 100%）与参数量增加（XS -> L）都能显著提升 PSNR。
- **主要优势**：极强的零样本泛化能力；无需 Pose 监督；缩放行为高度可预测。
- **主要局限**：对未观测区域缺乏生成能力（仅渲染模糊均值）；处理强动态场景时，动态物体会表现为模糊的“平均状态”。

## 6. 实用指南
- **开源情况**：代码已开源（https://github.com/compvis/rayder）。
- **训练细节**：
    - 必须使用 AdamW 优化器，且需要长期的线性 Warmup 预热，以防止相机内参预测发散。
    - 建议使用 6 帧输入、2 帧输出的采样配置。
    - 动态状态 Dropout 率建议设为 0.5。
- **迁移可能**：该架构可以直接迁移到任何需要从稀疏、非受控多视图图像序列中进行 3D 重建的任务。

## 7. 总结
- **核心思想**：统一架构 + 动态干扰吸收，将自监督 NVS 变为可缩放学习问题。
- **速记版pipeline**：
  1. 共享 Transformer 处理多视图图像；
  2. 预测姿态并预测干扰项以吸收动态信息；
  3. 采用随机顺序训练强制学习几何一致性；
  4. 推理时舍弃干扰项并利用并行注意力渲染。

**Key Findings:**

- Self-supervised novel view synthesis (NVS) remains challenging to scale, despite the abundance of video data, largely due to the brittleness of training on realistic videos and the hard-to-predict scaling behavior of multi-network system designs.
- We introduce RayDer, a unified, feed-forward transformer that consolidates camera estimation, scene reconstruction, and rendering into a single backbone, turning self-supervised NVS into a well-posed single-model scaling problem.
- Across multiple model sizes and orders of magnitude in data, RayDer exhibits clean power-law scaling with data and compute, and outperforms static-scene data mixtures.
- On a large number of benchmarks, RayDer achieves strong zero-shot open-set performance competitive with state-of-the-art supervised approaches.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31535v1)
- [arXiv](https://arxiv.org/abs/2605.31535v1)

---

<a id='2605.31508v1'></a>
## [Internalizing Temporal Consistency in Video Object-Centric Learning without Explicit Regularization](https://arxiv.org/abs/2605.31508v1)

**Authors:** Rongzhen Zhao, Zhiyuan Li, Juho Kannala, Joni Pajarinen

**Published:** 2026-05-29

**Categories:** cs.CV

**Abstract:**

Video Object-Centric Learning (OCL) aims to represent objects as \textit{slot} vectors and maintain their consistency across frames. Slot-Slot Contrastive (SSC) loss has become the cornerstone for state-of-the-art (SOTA) video OCL methods. While highly effective, SSC relies on one-to-one object correspondence across frames and introduces an extra loss. Following Occam's Razor, we propose a paradigm shift: temporal consistency is better enforced as an implicit model design rather than an explicit loss. To elegantly exclude SSC (\textbf{xSSC}), we introduce two quasi-zero-overhead synergistic mechanisms: (\textit{i}) Chrono-Channel Decomposition (CCD) structurally disentangles slot representations along the channel dimension into \textit{static} and \textit{dynamic} sub-spaces, serving as an empirically unified information bottleneck; (\textit{ii}) Cross-Temporal Reconstruction (CTR) stochastically reconstructs target features of either the current or previous time step by fusing current slots' static channels and target slots' dynamic channels, using a single standard OCL decoder with minor training adaptation. Thereby, the slot sets inherently learn temporal consistency by minimizing the standard reconstruction error alone. Extensive experiments show that integrating xSSC into leading baselines not only improves training efficiency but also establishes new SOTAs on video object discovery and recognition tasks. Furthermore, our PCA and gradient analyses confirm that objects' time-invariant semantics and time-variant kinematics are encoded into the proposed sub-spaces. Our source code, model checkpoints and training logs are provided on https://github.com/Genera1Z/xSSC.

**Analysis:**

这是一份关于论文《Occam’s Razor: Internalizing Temporal Consistency in Video Object-Centric Learning without Explicit Regularization》的深入技术分析。

### 1. 摘要翻译
视频物体中心学习（OCL）旨在将物体表示为槽（slot）向量并保持其在帧间的连续性。目前的SOTA方法严重依赖显式的槽-槽对比（SSC）损失。本文基于“奥卡姆剃刀”原则，提出一种范式转移：将时间一致性内化为隐含的模型设计，而非显式损失。为优雅地剔除SSC（xSSC），我们引入两个近乎零开销的协同机制：(i) 时序通道分解（CCD），在通道维度上将槽表示解耦为静态和动态子空间，作为统一的信息瓶颈；(ii) 跨时序重构（CTR），利用当前静态槽与目标动态槽通过标准OCL解码器进行随机重构。实验表明，xSSC不仅提升了训练效率，还在物体发现和识别任务上建立了新SOTA。

### 2. 方法动机分析
*   **驱动力**：试图消除视频OCL中对显式对比损失（如InfoNCE）的依赖，回归简洁的重构学习。
*   **痛点**：显式对比损失假设相邻帧存在稳定的“一对一”对应关系，在物体遮挡、消失或重现等动态复杂场景中，这种假设会成为脆弱的瓶颈，且引入了额外的计算开销和超参数。
*   **研究假设**：通过架构层面的结构化设计（即CCD和CTR），可以在不使用显式辅助损失的情况下，让模型自动学习时间一致性。

### 3. 方法设计详解
*   **流程总结**：
    1.  **Chrono-Channel Decomposition (CCD)**：将槽向量在通道维度拆分为两部分：静态（3/4通道，存物体身份/外观）和动态（1/4通道，存运动/位置）。
    2.  **Cross-Temporal Reconstruction (CTR)**：在训练时，随机选择目标帧（当前帧或前一帧）。
    3.  **融合与重构**：将当前帧的静态通道与目标帧的动态通道拼接，输入标准解码器重构当前帧特征。
*   **模型结构**：仅使用标准OCL解码器，无需额外的对比模块。
*   **算法意义**：通过强制模型利用“跨帧静态特征”来重构“目标特征”，迫使网络将外观与运动分离开来。如果静态特征不一致，则重构必然失败。

### 4. 方法对比分析
*   **本质区别**：从“通过显式损失正则化约束一致性”转变为“通过重构任务的结构约束隐含一致性”。
*   **创新贡献**：CCD结构提供了一种统一的、近乎零开销的特征解耦机制，证明了物体静态属性（低维流形）与动态属性的物理分布规律。
*   **适用场景**：高动态、频繁遮挡的视频场景。

### 5. 实验分析
*   **验证方法**：在MOVi-C/E和YTVIS-HQ数据集上，对比VideoSAUR、SlotContrast等方法。
*   **关键结论**：在保持甚至提升分割指标（ARI, mBO）的同时，显著提高了训练吞吐量和内存效率。梯度分析证实静态通道确实捕获了语义信息。
*   **主要局限**：对极度模糊的重叠实例可能重构受阻；对长程遮挡（超过随机帧窗口）的重识别仍有待优化。

### 6. 实用指南
*   **开源情况**：https://github.com/Genera1Z/xSSC。
*   **实现细节**：关键超参数为动态通道比例（建议为1/4）。训练时需结合相对时间嵌入（Relative Time Embedding）。
*   **迁移可能**：该方法可轻松集成到任何基于Slot Attention的视频架构中。

### 7. 总结
*   **核心思想**：通过通道解耦与跨时序重构，隐含实现时间一致性。
*   **速记版Pipeline**：
    1. 把槽位拆分成“长相”和“动作”两块。
    2. 训练时，让模型试着用这一帧的长相，去配合另一帧的动作。
    3. 如果拼得不对，模型就还原不了画面。
    4. 这种方式逼着模型自动记住物体，不需要再额外算对比损失。

**Key Findings:**

- Slot-Slot Contrastive (SSC) loss has become the cornerstone for state-of-the-art (SOTA) video OCL methods.
- Following Occam's Razor, we propose a paradigm shift: temporal consistency is better enforced as an implicit model design rather than an explicit loss.
- To elegantly exclude SSC (\textbf{xSSC}), we introduce two quasi-zero-overhead synergistic mechanisms: (\textit{i}) Chrono-Channel Decomposition (CCD) structurally disentangles slot representations along the channel dimension into \textit{static} and \textit{dynamic} sub-spaces, serving as an empirically unified information bottleneck; (\textit{ii}) Cross-Temporal Reconstruction (CTR) stochastically reconstructs target features of either the current or previous time step by fusing current slots' static channels and target slots' dynamic channels, using a single standard OCL decoder with minor training adaptation.
- Extensive experiments show that integrating xSSC into leading baselines not only improves training efficiency but also establishes new SOTAs on video object discovery and recognition tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31508v1)
- [arXiv](https://arxiv.org/abs/2605.31508v1)

---

<a id='2605.31460v1'></a>
## [On-Device Robotic Planning: Eliminating Inference Redundancy for Efficient Decision-Making](https://arxiv.org/abs/2605.31460v1)

**Authors:** Joonhee Lee, Hyunseung Shin, Hyunmi Kim, Pei Zhang, Jeonggil Ko

**Published:** 2026-05-29

**Categories:** cs.RO, eess.SY

**Abstract:**

Reasoning-based robotic policies using large language and vision-language models achieve strong semantic planning capabilities but mostly suffer from a high inference latency that limits practical real-time deployment. In this work, we observe that robotic reasoning workloads contain substantial temporal redundancy, where consecutive observations frequently produce identical actions and subgoals. Based on this insight, we present REIS, a human cognition inspired robotic decision-making framework that minimizes unnecessary reasoning while preserving semantic adaptability. REIS combines lightweight scene gating, KV-steered affordance routing, and deliberative reasoning to accelerate robotic control under embodied constraints. Experiments on ALFRED, and real-world robotic tasks demonstrate that REIS significantly suppresses reasoning overhead while maintaining competitive task performance.

**Analysis:**

这是一份针对论文《On-Device Robotic Planning: Eliminating Inference Redundancy for Efficient Decision-Making》的深度技术分析。

### 1. 摘要翻译
基于推理的机器人策略利用大型语言模型和视觉语言模型（VLM）实现了强大的语义规划能力，但通常受限于高推理延迟，难以在边缘设备上进行实时部署。本文观察到，机器人推理负载存在显著的时间冗余，即连续的观测往往产生相同的动作和子目标。基于此，我们提出了REIS，这是一个受人类认知启发、旨在最小化不必要推理同时保持语义适应性的机器人决策框架。REIS结合了轻量级场景门控、KV缓存导向的行动路由以及审慎推理，在具身约束下加速了机器人控制。在ALFRED及真实世界机器人任务上的实验表明，REIS在保持竞争性任务表现的同时，显著抑制了推理开销。

### 2. 方法动机分析
*   **驱动力**：解决VLM在机器人实时控制中因高延迟导致的“观测过时”问题。
*   **现有方法痛点**：现有方法倾向于将推理（慢）与动作生成（快）解耦，但这导致了“语义新鲜度”与“控制响应性”之间的脆弱权衡：更新慢则环境 stale，更新快则延迟极高。
*   **核心直觉**：环境中的视觉语义变化是稀疏的，大量的连续帧推理是冗余的。与其隔离推理，不如通过消除冗余来加速推理。

### 3. 方法设计详解
REIS的核心思想是双过程架构，将决策分为轻量级“系统一”（直觉）和审慎的“系统二”（推理）。
*   **系统一：快速直觉决策（Fast Intuitive Decision Making）**
    *   **EMA-HSVS（Ego-Motion Aware Head-Selective Vision Similarity）**：一种轻量级场景门控机制。它并非对所有视觉 token 进行计算，而是通过“贪婪校准”选择对几何变化最敏感的 Transformer 注意力头。在处理新帧时，先剔除机器人本体像素（Self-Mask），计算选定头的余弦相似度。只有当相似度低于预设阈值（即发生显著 structural 变化）时，才触发系统二。
    *   **KV-Steered Affordance Router**：利用 KV 缓存导向技术，将任务特定的 steering 向量注入模型，使其在不进行 autoregressive 推理的情况下，通过对预定义“是/否”问题的概率判断，快速确认子目标状态或维护当前动作。
*   **系统二：审慎推理（Deliberative Planning）**
    *   仅在系统一判定状态模棱两可或出现失效时激活。它基于离线合成的 steering tensors 来 bias 推理路径。通过对比“推理富集（+）”与“标签仅有（-）”的 KV 状态，计算 element-wise 差值得到 Steering Vectors（公式 1），用于引导系统一的轻量级判断。

### 4. 方法对比分析
*   **本质区别**：它不是简单的“快慢模型切换”，而是通过对 VLM 内部状态（KV cache）的干预，将复杂的生成式推理转化为轻量级的判定式查询。
*   **创新贡献**：设计了 EMA-HSVS 对视觉感知进行特异性压缩，并利用 KV Steering 对推理过程进行“预计算”，极大降低了边缘计算平台的内存与算力压力。
*   **适用场景**：适合所有基于 VLM/VLA 的长视野（Long-horizon）具身智能任务，特别是在计算资源受限的边缘嵌入式设备（如 Jetson Orin）上。

### 5. 实验分析
*   **验证方法**：在 ALFRED 模拟环境及真实世界的导航与操作任务中，对比“朴素（Naive）”推理与 REIS。
*   **关键结论**：在保持任务成功率仅微小下降（约 3-4%）的前提下，平均实现了 15 倍以上的速度提升。
*   **主要优势**：极低的延迟，使得实时避障和细粒度操作成为可能，将原先 10 秒级的失效检测降至 1 秒以内。
*   **主要局限**：目前主要针对高层指令调用，尚不能替代端到端的连续轨迹生成；对真实世界大模型微调数据仍有依赖。

### 6. 实用指南
*   **实现细节**：关键在于对 vision-encoder 头部的“贪婪搜索”，确保选出的 heads 对结构变化（而非光照等噪声）敏感。KV steering 向量的获取需要离线准备正负样本对。
*   **迁移可能**：该框架具有模型无关性。其门控逻辑可直接迁移到 CLIP/SigLIP 等视觉编码器，Affordance Router 的设计也可推广至任何基于 Transformer 的 VLA 模型。

### 7. 总结
*   **核心思想**：通过结构化视觉过滤与 KV 状态注入，消除具身智能推理的冗余性。
*   **速记版pipeline**：
    1.  **视觉过滤**：利用选定的注意力头检测关键视觉变化。
    2.  **快速路由**：若无变化，注入 steering 向量快速判定当前动作有效性。
    3.  **按需推理**：若判定为状态变化或失败，激活全量推理进行重规划。

**Key Findings:**

- Based on this insight, we present REIS, a human cognition inspired robotic decision-making framework that minimizes unnecessary reasoning while preserving semantic adaptability.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31460v1)
- [arXiv](https://arxiv.org/abs/2605.31460v1)

---

<a id='2605.31400v1'></a>
## [FSM-Net: An Efficient Frequency-Spatial Network for Real-World Deblurring](https://arxiv.org/abs/2605.31400v1)

**Authors:** Vinh-Thuan Ly

**Published:** 2026-05-29

**Categories:** cs.CV

**Abstract:**

Real-world image deblurring demands both high-fidelity restoration and computational efficiency, a balance existing methods often struggle to achieve. In this paper, we propose FSM-Net (Frequency-Spatial Multi-branch Network), a highly efficient solution that secured 2nd place in the NTIRE 2026 Challenge on Efficient Real-World Deblurring. FSM-Net pioneers a dual-domain approach: a novel Frequency Attention module explicitly recovers high-frequency structural details via FFT, while a Cross-Gated Vision E-Branchformer at the bottleneck captures global dependencies with linear complexity. To ensure robust convergence, we employ a progressive curriculum training strategy guided by a composite loss function (Multi-Scale Charbonnier, Structural Edge, and Frequency). Evaluated on the RSBlur benchmark, FSM-Net achieves an outstanding 33.144 dB PSNR with only 4.94M parameters and 159.35 GMACs (at 1920x1200 resolution). By effectively pushing the Pareto frontier of efficiency and quality, FSM-Net establishes a strong baseline for resource-constrained image restoration.

**Analysis:**

### 1. 摘要翻译
真实场景图像去模糊需要在高保真度修复与计算效率之间取得平衡，而现有方法往往难以实现。本文提出了FSM-Net（频率-空间多分支网络），这是一种高效解决方案，在NTIRE 2026高效真实场景去模糊挑战赛中荣获第二名。FSM-Net开创了一种双域方法：通过新型频率注意力模块利用FFT显式恢复高频结构细节，同时在瓶颈处通过交叉门控视觉E-Branchformer捕获线性复杂度的全局依赖。为确保鲁棒收敛，我们采用了由复合损失函数（多尺度Charbonnier、结构边缘及频率损失）引导的渐进式课程学习策略。在RSBlur基准测试上，FSM-Net仅需4.94M参数和159.35 GMACs（在1920x1200分辨率下）即实现了33.144 dB的优异PSNR。通过有效推高效率与质量的帕累托前沿，FSM-Net为资源受限的图像修复任务建立了强有力的基准。

### 2. 方法动机分析
- **驱动力**：运动模糊本质上是一种低通滤波过程，导致高频结构细节丢失；且高分辨率图像的计算开销巨大。
- **现有方法痛点**：传统卷积在空间域处理高频细节效果不佳；基于Transformer的自注意力机制在空间维度上呈二次复杂度，难以适配边缘计算的算力预算。
- **研究假设**：通过显式解耦空间特征与频率分量，并在频率域进行复数域调制，可以更高效地重建模糊带来的结构偏移与高频丢失。

### 3. 方法设计详解
- **总体架构**：基于NAFNet的U型分层结构。通过encoder-decoder进行特征层次化提取，瓶颈处引入E-Branchformer处理全局信息。
- **FSMBlock (核心创新)**：
  1. **特征拆分**：输入特征被拆分为两半：$X_1$（保持空间分支）和$X_2$（进入频率分支）。
  2. **频率注意力 (FAttn)**：对$X_2$执行rFFT，并与**复数权重** $W_c$ 进行元素级乘法。此复数操作同时调制**幅值（对比度）**与**相位（结构）**，用于修复由运动模糊导致的结构位移，最后通过irFFT转回空间域。
  3. **交叉门控**：将增强后的频率特征与空间分支$X_1$通过非线性门控机制融合，最后辅以通道注意力（SCA）进一步校准。
- **E-Branchformer**：
  - **局部CNN分支**：通过深度卷积捕获细粒度纹理。
  - **全局注意力分支**：采用**转置注意力（Transposed Attention）**，在通道维度计算协方差，将复杂度从 $O(H^2W^2C)$ 降至 $O(HWC^2)$，实现线性推理。
  - **交叉门控**：两分支特征互相生成Attention Gate，实现互增强。

### 4. 方法对比分析
- **本质区别**：与传统单纯空间域增强不同，FSM-Net引入了复数域的相位调制，这能更直接地解决非均匀运动模糊产生的鬼影问题。
- **创新贡献**：提出复数权重频率注意力机制及跨域信息交换策略，在极低参数量下实现了对全局和局部特征的高效捕捉。
- **适用场景**：适用于资源受限的实时图像修复任务，如移动端ISP处理、边缘视频去模糊。

### 5. 实验分析
- **验证方法**：在NTIRE 2026 Challenge的RSBlur数据集上进行测试，对比了多种SOTA模型。
- **关键结果**：FSM-Net在4.94M参数量下达到33.144 dB PSNR，推理速度极快（0.276s/图，TTA×4）。
- **优势与不足**：优势在于计算效率与高保真度细节恢复的平衡；不足在于对于极端非线性大尺度畸变，相比超大规模Transformer仍有提升空间。

### 6. 实用指南
- **实现细节**：
  - **训练策略**：必须采用“5阶段渐进式课程训练”，从低分辨率到全分辨率逐步引入不同损失函数。
  - **超参数**：$\alpha=0.999$ 的EMA是保证训练稳定性的关键。
  - **数据处理**：使用rFFT时需注意边界填充，确保相位调制准确。
- **迁移可能**：该复数域频率注意力模块可直接迁移至图像超分辨率（SR）或去噪任务，用以恢复纹理细节。

### 7. 总结
- **核心思想**：通过复数域相位调制和线性复杂度算子，实现轻量级去模糊。
- **速记版pipeline**：
  1. 拆分特征进入空间与频率分支；
  2. 对频率特征进行复数域相位与幅值校准；
  3. 两分支特征通过交叉门控机制融合；
  4. 使用转置注意力捕获长程依赖；
  5. 渐进式多阶段训练确保收敛。

**Key Findings:**

- In this paper, we propose FSM-Net (Frequency-Spatial Multi-branch Network), a highly efficient solution that secured 2nd place in the NTIRE 2026 Challenge on Efficient Real-World Deblurring.
- FSM-Net pioneers a dual-domain approach: a novel Frequency Attention module explicitly recovers high-frequency structural details via FFT, while a Cross-Gated Vision E-Branchformer at the bottleneck captures global dependencies with linear complexity.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31400v1)
- [arXiv](https://arxiv.org/abs/2605.31400v1)

---

<a id='2605.31343v1'></a>
## [Learning Terrain-Aware Whole-Body Control for Perceptive Legged Loco-Manipulation](https://arxiv.org/abs/2605.31343v1)

**Authors:** Sikai Guo, Yudong Zhong, Guoyang Zhao, Botao Dang, Zhihai Bi, Jun Ma

**Published:** 2026-05-29

**Categories:** cs.RO

**Abstract:**

Legged manipulators integrate exceptional terrain adaptability along with mobile manipulation capabilities, which make them highly promising for deployment in human-centric environments. By coordinating the control of both legs and arms, a whole-body controller can significantly expand the operational workspace of legged manipulators. However, many existing whole-body controllers primarily depend on proprioception and do not incorporate the critical exteroception required for effective terrain topology perception. This limitation can hinder their ability to adapt to varying environmental conditions and navigate complex terrains effectively. In this paper, we introduce TA-WBC, a terrain-aware whole-body control framework for legged manipulators, which features a novel RL-based unified policy tailored to whole-body loco-manipulation tasks in various terrains. Specifically, we employ a hybrid exteroception encoder to extract terrain features, providing an essential basis for the robot to proactively adapt posture and footholds. Furthermore, to facilitate stable cross-terrain loco-manipulation, we propose a novel end-effector sampling method based on the foot contact plane, decoupling manipulation target from base fluctuations. Moreover, a dual-policy distillation module is introduced to integrate expansive whole-body motion with terrain adaptability without catastrophic forgetting. The simulation and real-world experiments validate the robustness of our proposed controller, which leads to a larger reachable space, less tracking error, and reduced unexpected stumbles. This unified policy highlights the promising capabilities of legged manipulators in performing loco-manipulation tasks across complex terrains.

**Analysis:**

### 1. 摘要翻译
腿式机械臂因其出色的地形适应性和移动操作能力，在以人为中心的环境中极具应用前景。通过协调腿部和手臂的运动，全身控制器（WBC）能大幅扩展腿式机械臂的操作空间。然而，现有的WBC方法大多依赖本体感受，缺乏有效的地形拓扑感知，限制了其在复杂环境下的任务执行能力。为此，本文提出了TA-WBC（地形感知全身控制）框架。该框架采用基于强化学习（RL）的统一策略，集成了混合外感知编码器，通过提取地形特征实现姿态和落足点的主动适应。此外，提出了一种基于足端接触平面的末端执行器采样方法，解耦了基座扰动与操作目标。通过双策略蒸馏模块，TA-WBC在保持地形适应性的同时，实现了全身运动的无缝整合。仿真和实地实验验证了该方法在扩大操作空间、降低跟踪误差及减少意外碰撞方面的鲁棒性。

### 2. 方法动机分析
*   **驱动力**：解决腿式机械臂在复杂、非平坦地形（如阶梯、坡道）中同时进行精密操作的难题。
*   **痛点**：现有方法大多假设地形为平坦表面，依赖本体感受，面对地形起伏导致的基座波动，末端执行器跟踪精度极差；且缺乏环境感知能力，导致频繁的意外接触或碰撞。
*   **研究假设**：通过引入足端中心的地形感知，并将末端操作空间约束解耦至动态足端平面（FCP），可以实现基座扰动下的稳定末端跟踪与全身协调。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **输入感知**：本体感受信息（历史状态）+ 足端中心多环采样（外感知）。
    2.  **特征提取**：利用混合外感知编码器，对四个足端周围的地形高度进行卷积处理，输出精细化的局部环境表征。
    3.  **任务解耦**：构建基于足端接触平面（FCP）的采样坐标系，使末端操作目标相对于地面是平稳的。
    4.  **策略优化**：在双阶段框架下进行训练，通过蒸馏损失（Distillation Loss）将地形适应策略与全身协调策略合并至TA-WBC单一网络。
*   **关键技术**：
    *   **Hybrid Exteroceptive Encoder**：不再依赖全局高度图，而是对每个脚周围5个同心圆采样点进行卷积处理，利用平移不变性实现局部高分辨率感知。
    *   **FCP Sampling**：通过足端接触点的最小二乘拟合更新坐标系，从而确保末端执行器的目标是在一个“感知正确”的参考系下采样，抵消了基座姿态（尤其是Pitch角）带来的晃动误差。

### 4. 方法对比分析
*   **本质区别**：从传统的“基座中心参考系”切换为“足端局部参考系”，并将操作空间与行走空间通过感知蒸馏进行硬耦合。
*   **创新点**：足端多环采样策略（解决感知分辨率问题）与基于FCP的末端目标采样（解决基座干扰问题）。
*   **适用场景**：复杂、多层、地形不规则的非结构化环境下的移动操作。

### 5. 实验分析（精简版）
*   **验证方法**：在仿真环境中对比“盲WBC”、“解耦IK”与TA-WBC；在Real-world通过复杂阶梯抓取任务进行实测。
*   **结论**：TA-WBC在复杂地形下意外碰撞率显著降低（降低约400%+），且操作空间体积显著增加。
*   **局限**：对足端接触点的准确探测依赖于传感器数据的实时性，如果地形极其松软导致接触点难以判断，效果可能下降。

### 6. 实用指南
*   **实现细节**：
    *   **编码器结构**：利用1D卷积进行环形填充，能够显著提升训练收敛速度。
    *   **蒸馏策略**：在平坦地形训练全身协调策略（Expert A），复杂地形训练运动策略（Expert B），二者对齐损失（KL Divergence）是成功的关键。
*   **迁移可能**：可直接迁移至其他四足平台（如Anymal），只需调整对应的IK solver和电机接口。

### 7. 总结
*   **核心思想**：基于足端局部感知与接触平面的全身协同控制。
*   **速记版pipeline**：
    1. 提取以脚为中心的地形特征。
    2. 根据落足点动态调整操作坐标系。
    3. 通过两个专家策略蒸馏训练单一控制器。
    4. 采用PD控制与IK辅助实现最终动作执行。

**Key Findings:**

- In this paper, we introduce TA-WBC, a terrain-aware whole-body control framework for legged manipulators, which features a novel RL-based unified policy tailored to whole-body loco-manipulation tasks in various terrains.
- Furthermore, to facilitate stable cross-terrain loco-manipulation, we propose a novel end-effector sampling method based on the foot contact plane, decoupling manipulation target from base fluctuations.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31343v1)
- [arXiv](https://arxiv.org/abs/2605.31343v1)

---

<a id='2605.31321v1'></a>
## [Surface Constraint Policy for Learning Surface-Constrained and Dynamically Feasible Robot Skills](https://arxiv.org/abs/2605.31321v1)

**Authors:** Shuai Ke, Jiexin Zhang, Huan Zhao, Zhiao Wei, Yikun Guo, Jie Pan, Han Ding

**Published:** 2026-05-29

**Categories:** cs.RO

**Abstract:**

Diffusion-based imitation learning methods have driven rapid progress in robot dexterous manipulation tasks. However, they have limitations when applied to tasks that involve complex free-form surface constraints because of their lack of explicit surface geometry constraint modeling and the dynamic feasibility issue, resulting in stochastic action generation that fails to achieve reliable surface alignment and maintain stable contact. To address these limitations, we propose a novel surface constraint policy (SCP) for generating robot actions that satisfy free-form surface constraints on the basis of human demonstrations and real-time visual observations. First, the surface geometry constraint is encoded using a two-dimensional weighted Gaussian kernel function that is derived from demonstrations. Building on the encoded surface geometry constraints, the diffusion-based policy is used to infer task-level action intentions from multimodal sensory inputs, including visual observations and robot state feedback. These intentions are further transformed into surface-constrained dynamic movement primitives (DMPs) through a similarity-based action mapping method, thereby enabling smooth and compliant motion execution. The SCP achieves generation of structured surface geometric intent and dynamically admissible actions. The proposed method is validated on multiple surface manipulation tasks and compared with existing techniques. The experimental results demonstrate superior task success rates and contact stability under surface constraints.

**Analysis:**

### 1. 摘要翻译
基于扩散模型的模仿学习在机器人灵巧操作任务中取得了显著进展，但在涉及复杂自由曲面约束的任务中仍存在局限性。现有方法缺乏明确的曲面几何约束建模和动态可行性保证，导致生成的动作具有随机性，无法实现可靠的表面对齐和稳定的接触。为了解决这些问题，本文提出了一种新型表面约束策略（SCP）。该方法通过人类演示和实时视觉观测，生成满足自由曲面约束的机器人动作。首先，利用演示数据通过二维加权高斯核函数对曲面几何约束进行编码；在此基础上，利用扩散策略从多模态传感器输入中推断任务级动作意图；随后，通过一种基于相似度的映射方法将这些意图转换为表面约束下的动态运动基元（DMPs），从而实现平滑且合规的动作执行。实验表明，该方法在多个表面操作任务中显著提升了任务成功率和接触稳定性。

### 2. 方法动机分析
- **驱动力**：在诸如擦拭、抛光、超声扫描等复杂表面作业中，传统基于纯扩散策略的方法倾向于生成无约束的自由空间轨迹，忽略了接触力控制及几何形状的匹配。
- **现有痛点**：现有方法往往缺乏显式的几何约束建模，导致生成的轨迹与曲面之间存在对齐误差，且缺乏动态可行性，引发不平滑的加速度和不稳定的姿态。
- **核心直觉**：通过将曲面几何先验显式地编码到运动生成框架中，并利用动态运动基元（DMPs）确保执行层面的平滑度，可以将随机动作意图“引导”至符合物理现实的约束空间内。

### 3. 方法设计详解
- **Pipeline**：
    1. **曲面约束编码**：利用演示轨迹的末端执行器位置，通过本地加权回归（LWR）拟合二维加权高斯核函数，构建曲面几何状态表示 $S(u,v)$，作为几何先验。
    2. **动作意图推断**：采用基于Transformer的扩散模型，以图像序列、末端位姿及关节状态为条件，输出未经约束的动作意图轨迹 $A_t$。
    3. **相似性映射**：通过联合优化函数，将扩散模型输出的位置分量 $p_t^{pred}$ 投影至曲面 $S(u,v)$，并使用KL散度约束投影前后轨迹的“方向熵”，保持运动结构特征。
    4. **动态运动基元（DMPs）执行**：使用AL-DMP（位置）和Geo-DMP（方向），以弧长/测地线长度取代时间作为相位变量，实现位置与姿态在变速下的同步执行。

### 4. 方法对比分析
- **本质区别**：从传统的端到端“盲目”映射转变为“生成-投影-演化”的显式约束框架，将几何约束直接植入模型推理及后续的控制器中。
- **创新点**：提出了基于测地线的Geo-DMP与基于弧长的AL-DMP协同机制，并创新性地利用“方向熵”作为相似性度量来保证投影后的动力学一致性。
- **适用场景**：高曲率、强接触约束的工业精密作业，特别是不规则表面的清洁与抛光。

### 5. 实验分析
- **关键结论**：在飞机风挡清洁等高难度任务中，SCP取得了98%以上的成功率，远超传统的DP（失败率高）及ACP（仅63%）。
- **主要优势**：极大地降低了表面对齐误差（SAE），解决了接触过程中姿态偏移导致的“一侧抬起”问题，同时运动平滑度极高。
- **主要局限**：对演示数据的质量有较高依赖，且目前尚不支持完全未知的复杂环境即插即用。

### 6. 实用指南
- **开源/复现**：作者利用KUKA iiwa14机器人和Haption 6D设备采集数据。关键在于复现曲面编码模块中的LWR算法和基于测地线的DMP积分器。
- **实现细节**：观测视界 $T_o=2$，动作视界 $T_a=8$。在扩散模型训练中，使用1D卷积架构配合Transformer，建议重点调优公式(10)中的权重系数 $\lambda$。
- **迁移性**：该框架高度模块化，只需更换曲面几何表示模块，即可迁移到其他如焊接、打磨等工业接触作业任务中。

### 7. 总结
- **核心思想**：通过显式几何编码与相似性投影，实现约束感知的动态路径生成。
- **速记版pipeline**：
    1. 拟合曲面几何函数作为先验；
    2. 扩散模型推断原始动作意图；
    3. 相似性映射将意图投影至曲面；
    4. 测地线相位DMP确保平稳执行。

**Key Findings:**

- To address these limitations, we propose a novel surface constraint policy (SCP) for generating robot actions that satisfy free-form surface constraints on the basis of human demonstrations and real-time visual observations.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.31321v1)
- [arXiv](https://arxiv.org/abs/2605.31321v1)

---

