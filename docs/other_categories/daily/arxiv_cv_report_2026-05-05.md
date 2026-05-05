time: 20260505

# Arxiv Computer Vision Papers - 2026-05-05

## Executive Summary

以下是为您准备的每日报告执行摘要，涵盖了2026年5月4日arXiv计算机视觉领域的10篇新论文。

---

### **每日报告执行摘要：2026年5月4日 arXiv 计算机视觉论文**

**1. 主要主题与趋势**

本日论文呈现出几个清晰且相互关联的宏观趋势：

- **具身智能与视觉-语言-动作（VLA）模型的实用化浪潮**：超过三分之一的论文聚焦于将VLA模型从实验室推向真实世界部署。核心痛点包括：推理效率（Latent Bridge）、数据增强（Seeing Realism from Simulation）以及在复杂环境下的鲁棒性（MolmoAct2）。这标志着该领域正从“能否做到”向“如何可靠、高效地做到”转变。
- **多模态与跨模态学习的深化**：从单纯的融合走向更精细的交互与对齐。例如，LiDAR Teach, Radar Repeat 探讨了不同传感器模态间的知识蒸馏；ViewSAM 则处理了跨视角的语义对齐问题。
- **基础模型的效率与泛化性提升**：Linearizing Vision Transformer 通过测试时训练进行线性化，试图在保持Transformer性能的同时降低计算成本；Mamoda2.5 则探索了扩散Transformer（DiT）与专家混合（MoE）架构的结合，以增强统一多模态模型的能力。
- **几何与推理的回归**：在大量端到端学习之外，研究人员依然关注显式几何和结构化推理。AnchorD 使用因子图进行单目深度度量；Perceptual Flow Network 则强调查看-扫描-推理的主动感知流程，用于视觉推理。

**2. 突出亮点与创新论文**

- **最具实践价值**：**“LiDAR Teach, Radar Repeat”**。该工作提出了一种简单而强大的策略：利用高精度LiDAR在良好环境下训练一个雷达模型，使其在LiDAR失效的场景（如雨雾、镜面反射）中依然能保持可靠的导航能力。概念清晰，现实部署价值极高。
- **最具启发性**：**“Latent Bridge: Feature Delta Prediction for Efficient Dual-System VLA Model Inference”**。该论文提出“特征增量预测”而非完整重新计算特征，从而在快-慢双系统（System 1 & 2）推理框架下大幅降低计算开销。这是对VLA模型实际部署中“思考速度”瓶颈的一个巧妙且高效的解决方案。
- **最前沿架构探索**：**“Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE”**。将新兴的扩散Transformer（DiT）与成熟的专家混合（MoE）架构结合，代表了下一代统一多模态模型在扩展性与性能上的重要探索方向。

**3. 新兴研究方向与技术**

- **数据增强与仿真到现实的特定方法**：**“Seeing Realism from Simulation”** 专注于将仿真生成的VLA数据转移到真实场景的视频风格，而非传统的图像级域适应。这为VLA训练数据匮乏问题提供了新的、更高效的解决思路。
- **测试时训练用于模型线性化**：**“Linearizing ViT with Test-Time Training”** 提出了一种动态适应方案，即在推理过程中针对单个样本微调模型以使其线性化。这可能为简化大规模视觉模型部署指明了新路径。
- **弱监督下的跨视角多目标跟踪**：**“ViewSAM”** 利用SAM模型和跨模态语义学习，在只有粗略标签的数据上实现了跨视角的多目标跟踪，降低了数据标注成本。

**4. 值得全文精读的论文推荐**

- **最高优先级**：
    - **(2) LiDAR Teach, Radar Repeat** （实际系统设计的绝佳范例）
    - **(6) Latent Bridge** （解决VLA推理效率的关键创新）
    - **(1) MolmoAct2** （了解VLA模型在真实世界部署的现状与挑战）

- **次高优先级**：
    - **(4) Unified Map Prior Encoder** （若您专注于自动驾驶中的地图与规划）
    - **(9) Mamoda2.5** （若您对多模态大模型架构前沿感兴趣）
    - **(3) Linearizing ViT with Test-Time Training** （若您关注Transformer推理效率的理论与实践）

**总结**：本日论文表明，计算机视觉领域正加速拥抱“部署驱动”的研究范式。最具影响力的工作并非来自全新的模型架构，而是针对现有VLA系统的效率、鲁棒性和数据瓶颈提出了巧妙、具有工程智慧的解决方案。对于忙碌的研究人员，**优先阅读“LiDAR 教，雷达学”** 和 **“Latent Bridge”**，将能快速把握当前领域最务实的进步脉搏。

---

## Table of Contents

1. [MolmoAct2: Action Reasoning Models for Real-world Deployment](#2605.02881v1)
2. [LiDAR Teach, Radar Repeat: Robust Cross-Modal Navigation in Degenerate and Varying Environments](#2605.02809v1)
3. [Linearizing Vision Transformer with Test-Time Training](#2605.02772v1)
4. [Unified Map Prior Encoder for Mapping and Planning](#2605.02762v1)
5. [Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](#2605.02757v1)
6. [Latent Bridge: Feature Delta Prediction for Efficient Dual-System Vision-Language-Action Model Inference](#2605.02739v1)
7. [Perceptual Flow Network for Visually Grounded Reasoning](#2605.02730v1)
8. [AnchorD: Metric Grounding of Monocular Depth Using Factor Graphs](#2605.02667v1)
9. [Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE](#2605.02641v1)
10. [ViewSAM: Learning View-aware Cross-modal Semantics for Weakly Supervised Cross-view Referring Multi-Object Tracking](#2605.02638v1)

---

## Papers

<a id='2605.02881v1'></a>
## [MolmoAct2: Action Reasoning Models for Real-world Deployment](https://arxiv.org/abs/2605.02881v1)

**Authors:** Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, Shanli Xing, Jaemin Cho, Jae Sung Park, Ainaz Eftekhar, Peter Sushko, Karen Farley, Angad Wadhwa, Cole Harrison, Winson Han, Ying-Chun Lee, Eli VanderBilt, Rose Hendrix, Suveen Ellawela, Lucas Ngoo, Joyce Chai, Zhongzheng Ren, Ali Farhadi, Dieter Fox, Ranjay Krishna

**Published:** 2026-05-04

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models aim to provide a single generalist controller for robots, but today's systems fall short on the criteria that matter for real-world deployment. Frontier models are closed, open-weight alternatives are tied to expensive hardware, reasoning-augmented policies pay prohibitive latency for their grounding, and fine-tuned success rates remain below the threshold for dependable use. We present MolmoAct2, a fully open action reasoning model built for practical deployment, advancing its predecessor along five axes. We introduce MolmoER, a VLM backbone specialized for spatial and embodied reasoning, trained on a 3.3M-sample corpus with a specialize-then-rehearse recipe. We release three new datasets spanning low-to-medium cost platforms, including MolmoAct2-BimanualYAM, 720 hours of teleoperated bimanual trajectories that constitute the largest open bimanual dataset to date, together with quality-filtered Franka (DROID) and SO100/101 subsets. We provide OpenFAST, an open-weight, open-data action tokenizer trained on millions of trajectories across five embodiments. We redesign the architecture to graft a flow-matching continuous-action expert onto a discrete-token VLM via per-layer KV-cache conditioning. Finally, we propose MolmoThink, an adaptive-depth reasoning variant that re-predicts depth tokens only for scene regions that change between timesteps, retaining geometric grounding at a fraction of prior latency. In the most extensive empirical study of any open VLA to date, spanning 7 simulation and real-world benchmarks, MolmoAct2 outperforms strong baselines including Pi-05, while MolmoER surpasses GPT-5 and Gemini Robotics ER-1.5 across 13 embodied-reasoning benchmarks. We release model weights, training code, and complete training data. Project page: https://allenai.org/blog/molmoact2

**Analysis:**

以下是对《MolmoAct2: Action Reasoning Models for Real-World Deployment》论文的方法分析：

### 1. 摘要翻译
视觉-语言-动作 (VLA) 模型旨在为机器人提供通用控制器，但现有系统难以满足实际部署的需求。现有前沿模型不开源、开放权重模型往往绑定昂贵硬件、推理增强策略存在严重的延迟问题，且微调成功率较低。我们提出了 MolmoAct2，这是一种用于实际部署的完全开源的动作推理模型，它在五个方面超越了其前身 MolmoAct。我们引入了 Molmo2-ER，这是一种专门用于空间和具身推理的 VLM 主干模型；发布了涵盖低至中成本平台的三种新数据集，包括最大的开源双臂操作数据集；提供了 OpenFAST 分词器；重新设计了架构，将流匹配连续动作专家通过逐层 KV 缓存条件注入到离散令牌 VLM 中；最后，提出了 MolmoAct2-Think，一种自适应深度推理变体，仅对场景变化区域重预测深度标记，显著降低延迟。

### 2. 方法动机分析
*   **驱动力**：解决 VLA 模型从学术基准走向现实世界部署的“最后一公里”问题（如延迟、泛化能力、开源可及性）。
*   **痛点**：现有 VLA 模型推理延迟过高（因需生成过多 Reasoning 令牌）、通用模型缺乏对机器人任务至关重要的几何/空间理解，且由于缺乏开源数据导致社区难以进行适配性微调。
*   **核心直觉**：通过“ specialize-then-rehearse”（特化后复习）策略构建更强的具身骨干模型，并结合高效的动作专家连接（KV Cache 交互）和自适应深度推理，实现低延迟与高精度的平衡。

### 3. 方法设计详解
*   **流水线流程**：
    1.  **骨干预训练**：基于 Molmo2 进行空间具身数据特化训练，获得 Molmo2-ER 主干。
    2.  **动作分词**：使用 OpenFAST 将连续轨迹映射为紧凑的离散动作令牌，支持多种动作表征。
    3.  **架构连接（核心）**：将 DiT 风格的动作专家模块“嫁接”在 VLM 上，动作专家通过逐层交叉注意力（Cross-Attention）实时读取 VLM 的 KV 缓存，从而获得深度视觉语义特征，而非仅基于最终隐藏状态。
    4.  **自适应推理（MolmoAct2-Think）**：引入“深度标记预测”。系统根据 RGB 补丁的余弦相似度判断场景变化，仅对变化区域重预测深度码，大幅减少计算开销。
*   **模型结构**：由 VLM 主干和动作专家头组成。专家头具有与 VLM 相同的层数，实现了深度的特征交互。
*   **关键公式**：$L_{flow} = \mathbb{E}_{a,\epsilon,t} [\|m \odot (f_\theta(x_t, t, c) - u^\star)\|_2^2]$。这利用流匹配技术（Flow Matching）在离散令牌控制的基础上，学习连续动作的平滑轨迹生成。

### 4. 方法对比分析
*   **本质区别**：不同于常规 VLA 仅在最终层进行特征投影，MolmoAct2 采用“逐层 KV 连接”，使动作专家能获取模型全层的空间感知信息。
*   **创新点**：
    1.  **逐层 KV 缓存交互**：实现动作头与视觉语言主干的高效低延迟耦合。
    2.  **自适应深度推理**：通过缓存复用机制实现空间几何推理的计算量动态优化。
*   **适用场景**：适用于各种结构差异较大的低中成本机器人平台，特别是在需要高响应速度的实时控制任务中。

### 5. 实验分析
*   **验证方法**：在 7 个环境基准（如 LIBERO、RoboEval）及多个真实世界机器人平台进行测试。
*   **结论**：在 13 项 embodied-reasoning 基准中，Molmo2-ER 平均得分领先于 GPT-5 和 Gemini Robotics ER-1.5；在 Libero 任务上，MolmoAct2-Think 达到 98.1% 的成功率，推理帧率较原版大幅提升。

### 6. 实用指南
*   **开源情况**：完全开源，含权重、代码和数据集（BimanualYAM 等）。
*   **关键实现**：必须重视 `OpenFAST Tokenizer` 的配置，以及动作专家头与 VLM 主干的 KV 缓存适配（adapter layers）。
*   **迁移方案**：通过其提供的 Embodiment-specific fine-tuning 流程，在特定机器人上进行 50K-100K 步的少量样本微调即可适配新形态。

### 7. 总结
*   **核心思想**：通过逐层 KV 连接与自适应深度推理，实现高精度且低延迟的具身动作控制。
*   **速记版 Pipeline**：
    1. 训练空间特化主干模型；
    2. 对动作轨迹进行分词压缩；
    3. 接入 KV 缓存动作专家；
    4. 对动态场景区域进行自适应深度推理。

**Key Findings:**

- We present MolmoAct2, a fully open action reasoning model built for practical deployment, advancing its predecessor along five axes.
- We introduce MolmoER, a VLM backbone specialized for spatial and embodied reasoning, trained on a 3.3M-sample corpus with a specialize-then-rehearse recipe.
- We release three new datasets spanning low-to-medium cost platforms, including MolmoAct2-BimanualYAM, 720 hours of teleoperated bimanual trajectories that constitute the largest open bimanual dataset to date, together with quality-filtered Franka (DROID) and SO100/101 subsets.
- Finally, we propose MolmoThink, an adaptive-depth reasoning variant that re-predicts depth tokens only for scene regions that change between timesteps, retaining geometric grounding at a fraction of prior latency.
- In the most extensive empirical study of any open VLA to date, spanning 7 simulation and real-world benchmarks, MolmoAct2 outperforms strong baselines including Pi-05, while MolmoER surpasses GPT-5 and Gemini Robotics ER-1.5 across 13 embodied-reasoning benchmarks.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02881v1)
- [arXiv](https://arxiv.org/abs/2605.02881v1)

---

<a id='2605.02809v1'></a>
## [LiDAR Teach, Radar Repeat: Robust Cross-Modal Navigation in Degenerate and Varying Environments](https://arxiv.org/abs/2605.02809v1)

**Authors:** Renxiang Xiao, Yichen Chen, Yuanfan Zhang, Qianyi Shao, Yushuai Chen, Yuxuan Han, Yunjiang Lou, Liang Hu

**Published:** 2026-05-04

**Categories:** cs.RO

**Abstract:**

Long-term autonomy requires robust navigation in environments subject to dynamic and static changes, as well as adverse weather conditions. Teach-and-Repeat (T\&R) navigation offers a reliable and cost-effective solution by avoiding the need for consistent global mapping; however, existing T\&R systems lack a systematic solution to tackle various environmental variations such as weather degradation, ephemeral dynamics, and structural changes. This work proposes LTR$^2$, the first cross-modal, cross-platform LiDAR-Teach-and-Radar-Repeat system that systematically addresses these challenges. LTR$^2$ leverages LiDAR during the teaching phase to capture precise structural information under normal conditions and utilizes 4D millimeter-wave radar during the repeating phase for robust operation under environmental degradations. To align sparse and noisy forward-looking 4D radar with dense and accurate omnidirectional 3D LiDAR data, we introduce a Cross-Modal Registration (CMR) network that jointly exploits Doppler-based motion priors and the physical laws governing LiDAR intensity and radar power density. Furthermore, we propose an adaptive fine-tuning strategy that incrementally updates the CMR network based on localization errors, enabling long-term adaptability to static environmental changes without ground-truth labels. We demonstrate that the proposed CMR network achieves state-of-the-art cross-modal registration performance on the open-access dataset. Then we validate LTR$^2$ across three robot platforms over a large-scale, long-term deployment (40+ km over 6 months), including challenging conditions such as nighttime smoke. Experimental results and ablation studies demonstrate centimeter-level accuracy and strong robustness against diverse environmental disturbances, significantly outperforming existing approaches.

**Analysis:**

作为计算机视觉与机器人领域的专家，我对这篇题为《LiDAR Teach, Radar Repeat: Robust Cross-Modal Navigation in Degenerate and Varying Environments》的论文分析如下：

### 1. 核心贡献摘要
该论文提出了首个跨模态、跨平台的“激光雷达示教-毫米波雷达复现”（LTR$^2$）导航系统，旨在解决传统导航系统在极端天气、动态环境及长期环境变化下的鲁棒性瓶颈。通过结合激光雷达的高精度几何结构感知与4D毫米波雷达的全天候环境适应性，该系统实现了在无需持续全局建图条件下的长期高精度自主导航。

### 2. 关键创新点与方法论
*   **跨模态配准网络（CMR Network）：** 该方法的核心在于处理LiDAR（稠密、高精度）与4D雷达（稀疏、噪点多、非几何结构）之间的巨大模态差异。作者创新性地引入了基于**多普勒运动先验**的物理一致性约束，并将LiDAR强度与雷达功率密度相结合，实现了不同模态数据的有效特征对齐。
*   **自适应在线微调策略：** 为应对长期运行中的静态环境演变，系统引入了一种基于定位误差反馈的无监督微调机制。这使得系统能够在缺乏真值（Ground Truth）的情况下，随着时间推移自动更新配准模型，体现了极强的环境适应性。
*   **跨平台通用性：** 论文证明了该方法不仅限于单一平台，通过40公里、6个月的实地部署验证了其在大范围复杂场景下的普适性。

### 3. 对领域的潜在影响
*   **突破T&R系统的局限性：** “示教-复现”（T&R）模式以往多依赖单一传感器，极易因天气改变而失效。该研究通过跨模态解耦（LiDAR负责“存”，Radar负责“行”），为自动驾驶和移动机器人提供了一种低成本、高鲁棒性的全天候方案。
*   **多传感器融合的新范式：** 证明了即便模态异质性极高，通过深度学习结合物理规律（Physics-informed ML），依然能够实现超越单一模态的感知性能，这为未来传感器融合研究提供了重要借鉴。

### 4. 相关受益领域与应用
*   **全天候自动驾驶：** 特别是在雾天、烟雾、夜间等视觉和LiDAR性能退化的场景中。
*   **仓储与工业物流机器人：** 在结构可能发生细微变化（如货架移动、堆放改变）的仓库环境中，该系统的长期适应性具有巨大价值。
*   **地下与矿区作业：** 这些环境光照条件极差且存在粉尘，雷达的穿透特性配合LiDAR的初始高精度建图，是理想的导航方案。

### 5. 可推断的潜在限制
*   **计算开销与延迟：** 尽管提出了高效的CMR网络，但同时运行LiDAR感知、Radar处理以及在线微调模型对边缘侧计算资源（如嵌入式GPU）提出了较高要求。
*   **环境演变的边界：** 虽然系统能适应环境的静态演变，但如果环境发生灾难性结构突变（如大规模建筑重构），该系统可能仍需重新示教。
*   **对雷达性能的依赖：** 系统高度依赖4D毫米波雷达的质量。若雷达数据在多径反射极其严重的室内复杂金属环境中出现严重鬼影，其配准精度可能会面临挑战。

**专家总结：**
这篇论文的趣味性在于它巧妙地利用了“异构冗余”——利用LiDAR的高结构信息作为“记忆锚点”，通过深度学习将低成本但鲁棒的雷达对齐到该锚点上。这种**“以空间结构为主导，以物理特性为桥梁”**的思路，是解决计算机视觉中跨模态配准难题的一个非常精彩的范例，对于推动机器人从实验室走向真正“全天候、全季节”的落地具有重要意义。

**Key Findings:**

- To align sparse and noisy forward-looking 4D radar with dense and accurate omnidirectional 3D LiDAR data, we introduce a Cross-Modal Registration (CMR) network that jointly exploits Doppler-based motion priors and the physical laws governing LiDAR intensity and radar power density.
- Furthermore, we propose an adaptive fine-tuning strategy that incrementally updates the CMR network based on localization errors, enabling long-term adaptability to static environmental changes without ground-truth labels.
- We demonstrate that the proposed CMR network achieves state-of-the-art cross-modal registration performance on the open-access dataset.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02809v1)
- [arXiv](https://arxiv.org/abs/2605.02809v1)

---

<a id='2605.02772v1'></a>
## [Linearizing Vision Transformer with Test-Time Training](https://arxiv.org/abs/2605.02772v1)

**Authors:** Yining Li, Dongchen Han, Zeyu Liu, Hanyi Wang, Yulin Wang, Gao Huang

**Published:** 2026-05-04

**Categories:** cs.CV

**Abstract:**

While linear-complexity attention mechanisms offer a promising alternative to Softmax attention for overcoming the quadratic bottleneck, training such models from scratch remains prohibitively expensive. Inheriting weights from pretrained Transformers provides an appealing shortcut, yet the fundamental representational gap between Softmax and linear attention prevents effective weight transfer. In this work, we address this conversion challenge from two perspectives: architectural alignment and representational alignment. We identify Test-Time Training (TTT) as a linear-complexity architecture whose two-layer dynamic formulation is structurally aligned with Softmax attention, enabling direct inheritance of pretrained attention weights. To further align representational properties, including key shift-invariance and locality, we introduce key instance normalization and a lightweight locality enhancement module. We validate our approach by linearizing Stable Diffusion 3.5 and introduce SD3.5-T$^5$ (Transformer To Test Time Training). With only 1 hour of fine-tuning on 4$\times$H20 GPUs, SD3.5-T$^5$ achieves comparable text-to-image quality to the fine-tuned Softmax model, while accelerating inference by 1.32$\times$ and 1.47$\times$ at 1K and 2K resolutions.

**Analysis:**

### 1. 摘要翻译
尽管线性复杂度注意力机制为克服Softmax注意力的二次算力瓶颈提供了潜力，但从头训练这类模型代价高昂。直接继承预训练Transformer的权重虽然具备吸引力，但Softmax与线性注意力之间的巨大表征差异阻碍了有效的权重迁移。本文从结构对齐与表征对齐两个维度解决了这一转换挑战：我们识别出测试时间训练（TTT）架构在结构上与Softmax注意力兼容，支持权重直接继承；此外，我们引入了实例归一化（Instance Normalization）和轻量级局部增强模块，以对齐关键的表征属性（如平移不变性和局部性）。我们在Stable Diffusion 3.5上进行了验证，仅需1小时的微调，SD3.5-T5在保持文本到图像生成质量的同时，在1K和2K分辨率下分别实现了1.32倍和1.47倍的推理加速。

### 2. 方法动机分析
*   **驱动力**：在不进行大规模重训练的前提下，将高性能预训练Transformer（尤其是大模型）高效转化为线性复杂度的模型，以满足高分辨率或长序列推理的实时性需求。
*   **现有痛点**：传统线性注意力（基于核的方法）与Softmax注意力的表征空间存在本质差异（即单层动态线性变换vs两层动态非线性MLP），导致简单的权重继承效果极差，且现有的蒸馏方法极其复杂。
*   **研究假设**：如果选择在结构上能够模拟Softmax注意力两层动态非线性特征的架构，并补齐其在“平移不变性”和“局部性”上的缺失，就能实现无需大规模蒸馏的快速权重迁移。

### 3. 方法设计详解
*   **核心 Pipeline**：
    1.  **结构适配**：用TTT层替换传统的Softmax注意力层。TTT利用两层MLP动态建模，在结构上完美对齐了Softmax attention中 $qK^T$ 经过非线性变换后的表达能力。
    2.  **表征对齐（平移不变性）**：Softmax注意力天生具备对Key偏移的不变性。作者发现预训练模型中Key存在较大的偏置，直接将TTT应用于此会导致梯度爆炸。通过在Key上应用**Instance Normalization**，迫使Key中心化，解决了训练不稳定性。
    3.  **表征对齐（局部性增强）**：线性模型往往缺乏局部建模能力。作者引入**Depthwise Convolution (DWC)** 在Q和K上注入局部偏置，使模型能通过局部窗口的Key组合来预测Value，从而弥补了TTT全局建模带来的局部细节缺失。
*   **算法本质**：TTT将序列建模转化为一个动态在线学习问题，其内层循环通过梯度下降更新权重，这种结构天然承载了Softmax注意力那种随着Input变化而动态改变权重映射的能力。

### 4. 方法对比分析
*   **本质区别**：与CLEAR、LiT等方法不同，本文不依赖蒸馏或复杂的层归一化微调，而是通过**结构化重构**（使线性模型具备非线性表达）和**表征校准**（通过归一化和卷积对齐原有分布），实现“即插即用”式的全权重继承。
*   **创新贡献**：首次揭示了预训练Transformer中Key的“非零均值偏置”对线性化训练稳定性的破坏作用，并利用轻量级组件解决了这一难题。
*   **适用场景**：极高分辨率生成任务（如SD3.5），以及对推理算力要求极高、但又不能放弃原模型生成效果的场景。

### 5. 实验分析
*   **验证方法**：在DeiT（分类）和DiT/Stable Diffusion（生成）上验证。
*   **关键结论**：相比于全量重新训练，本文方法仅需不到1%的训练步数（或约1小时）即可实现接近原模型的效果，同时在高分辨率下展现出显著的线性加速优势。
*   **优势**：训练极其高效，推理速度快，无需知识蒸馏。
*   **局限**：模型参数量略有增加（引入了少量的TTT参数），且对于极复杂的分布，性能仍存在微小损耗（由FID指标略高于原模型看出）。

### 6. 实用指南
*   **开源情况**：已发布相关工作（参考ViT3和文中引用），代码实现建议参考文中提到的结构设计（TTT-SwiGLU）。
*   **关键细节**：
    *   **LR Multiplier**：对于新初始化的TTT参数，需要使用比预训练基座大20倍的学习率。
    *   **归一化处理**：必须保留Key的减均值操作（Instance Norm），否则必崩。
    *   **适配器**：将注意力块替换为TTT时，保持Q/K/V的投影层不变，仅在内部构建内层模型。

### 7. 总结
*   **核心思想**：通过结构对齐与表征校准，将线性模型转化为高性能预训练Transformer的等价变体。
*   **速记版pipeline**：
    1.  用两层MLP构成的TTT模块替换原注意层。
    2.  在Keys上应用实例归一化以确保分布中心化。
    3.  在Query和Key输入端加入深度卷积注入局部性。
    4.  继承原模型权重并以高学习率微调TTT参数。

**Key Findings:**

- To further align representational properties, including key shift-invariance and locality, we introduce key instance normalization and a lightweight locality enhancement module.
- We validate our approach by linearizing Stable Diffusion 3.5 and introduce SD3.5-T$^5$ (Transformer To Test Time Training).

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02772v1)
- [arXiv](https://arxiv.org/abs/2605.02772v1)

---

<a id='2605.02762v1'></a>
## [Unified Map Prior Encoder for Mapping and Planning](https://arxiv.org/abs/2605.02762v1)

**Authors:** Zongzheng Zhang, Sizhe Zou, Guantian Zheng, Zhenxin Zhu, Yu Gao, Guoxuan Chi, Shuo Wang, Yuwen Heng, Zhigang Sun, Yiru Wang, Hao Sun, Chao Ma, Zhen Li, Anqing Jiang, Hao Zhao

**Published:** 2026-05-04

**Categories:** cs.CV

**Abstract:**

Online mapping and end-to-end (E2E) planning in autonomous driving remain largely sensor-centric, leaving rich map priors, including HD/SD vector maps, rasterized SD maps, and satellite imagery, underused because of heterogeneity, pose drift, and inconsistent availability at test time. We present UMPE, a Unified Map Prior Encoder that can ingest any subset of four priors and fuse them with BEV features for both mapping and planning. UMPE has two branches. The vector encoder pre-aligns HD/SD polylines with a frame-wise SE(2) correction, encodes points via multi-frequency sinusoidal features, and produces polyline tokens with confidence scores. BEV queries then apply cross-attention with confidence bias, followed by normalized channel-wise gating to avoid length imbalance and softly down-weight uncertain sources. The raster encoder shares a ResNet-18 backbone conditioned by FiLM with scaling and shift at every stage, performs SE(2) micro-alignment, and injects priors through zero-initialized residual fusion, so the network starts from a do-no-harm baseline and learns to add only useful prior evidence. A vector-then-raster fusion order reflects the inductive bias of geometry first, appearance second. On nuScenes mapping, UMPE lifts MapTRv2 from 61.5 to 67.4 mAP (+5.9) and MapQR from 66.4 to 71.7 mAP (+5.3). On Argoverse2, UMPE adds +4.1 mAP over strong baselines. UMPE is compositional: when trained with all priors, it outperforms single-prior models even when only one prior is available at test time, demonstrating powerset robustness. For E2E planning with the VAD backbone on nuScenes, UMPE reduces trajectory error from 0.72 to 0.42 m L2 on average (-0.30 m) and collision rate from 0.22% to 0.12% (-0.10%), surpassing recent prior-injection methods. These results show that a unified, alignment-aware treatment of heterogeneous map priors yields better mapping and better planning.

**Analysis:**

## 1. 摘要翻译
在线地图构建与端到端（E2E）自动驾驶规划在很大程度上仍以传感器为中心，导致丰富的地图先验信息（HD/SD矢量地图、栅格化SD地图和卫星影像）因异构性、位姿漂移及测试时可用性不一致而未被充分利用。我们提出了 UMPE，一个统一的地图先验编码器，可摄入四种先验信息的任意子集，并将其与BEV特征融合，用于地图构建和规划。UMPE 包含两个分支：矢量编码器通过帧级 SE(2) 校正预对齐 HD/SD 多段线，利用多频率正弦特征编码点，并生成带有置信度分数的矢量标记。BEV 查询随后应用带置信度偏差的交叉注意力机制，辅以归一化通道门控，以避免长度不平衡并柔性地抑制不确定来源。栅格编码器共享一个由 FiLM（每一层进行缩放/偏移）调节的 ResNet-18 主干，执行 SE(2) 微对齐，并通过零初始化残差融合注入先验，使网络从“不伤害”基准开始，仅学习添加有用的先验证据。矢量优先于栅格的融合顺序反映了“先几何，后外观”的归纳偏置。在 nuScenes 地图构建任务上，UMPE 将 MapTRv2 的 mAP 从 61.5 提升至 67.4 (+5.9)，MapQR 从 66.4 提升至 71.7 (+5.3)。在 Argoverse2 上，UMPE 在强基线上额外提升了 +4.1 mAP。UMPE 具有组合性：当使用所有先验进行训练时，即使测试时仅使用一种先验，其性能也优于单先验模型，展现了强大的幂集鲁棒性。在 E2E 规划（VAD 主干，nuScenes）中，UMPE 将轨迹误差从 0.72 降至 0.42 m L2 (平均 -0.30 m)，碰撞率从 0.22% 降至 0.12% (-0.10%)，超越了最近的先验注入方法。这些结果表明，对异构地图先验进行统一的、对齐感知的处理能带来更好的地图构建和规划效果。

## 2. 方法动机分析
- **驱动力**：解决自动驾驶中多源地图先验（向量/栅格）异构、位姿不准以及测试时缺失的问题。
- **痛点**：当前方法通常仅支持特定组合或单一源，面对测试时传感器或地图可用性变化时鲁棒性差。
- **假设**：通过“先几何（向量）、后外观（栅格）”的统一编码框架，并引入“源丢失（SourceDropout）”训练策略，可以实现对任意先验子集的组合式鲁棒使用。

## 3. 方法设计详解
- **核心Pipeline**：
    1. **矢量编码分支**：采用 SE(2) 对齐纠偏，用正弦特征编码多段线点，生成置信度标记。通过带置信度偏差的交叉注意力机制与 BEV 查询交互。
    2. **栅格编码分支**：共享 ResNet-18，通过 FiLM 调制实现源感知，执行 SE(2) 微对齐。
    3. **零初始化残差融合**：利用 $X_{UMPE} = LN(\bar{Y}) + \alpha WLN(\bar{Z})$ 公式，确保初始阶段对 BEV 特征无影响（do-no-harm），随着 $\alpha$ 增加，逐步注入先验信息。
- **关键设计**：
    - **SourceDropout**：训练时随机禁用部分先验源，增强模型在测试时的“幂集”组合适应能力。
    - **通道门控（Gated Fusion）**：基于 presenza-normalized（存在归一化）softmax 门控，当某源缺失时，自动关闭该通道。

## 4. 方法对比分析
- **本质区别**：从传统的“固定多源融合”转变为“即插即用的任意子集融合”，实现了对不同地图源的动态选择。
- **创新点**：引入“先几何后外观”的层级融合顺序，以及零初始化残差路径，显著提升了与原传感器特征的兼容性。
- **适用场景**：适用于城市驾驶环境，既能利用精细的 HD 矢量图，也能在矢量缺失时依赖卫星影像进行补全。

## 5. 实验分析（精简版）
- **验证方法**：在 nuScenes 和 Argoverse 2 数据集上，将 UMPE 插入 MapTRv2/MapQR (映射) 和 VAD (规划) 基线。
- **关键结果**：在缺失 HD 地图的测试情境下，UMPE 的表现依然超过专门针对该单一源训练的模型，证明了协同训练的价值。
- **优势**：显著提升了几何精确度（矢量先验）和环境纹理感知（栅格先验）。
- **局限**：对计算资源有一定要求，虽然不显著，但在极端嵌入式平台可能仍需进一步剪枝。

## 6. 实用指南
- **开源地址**：https://github.com/Ethan-Zheng136/UMPE
- **迁移建议**：该方法模块化程度极高，可直接将 Vector/Raster 编码器作为辅助分支接入任何 BEV-based 的自动驾驶网络中。
- **关键超参数**：$\alpha$ 的调度计划（从0增加到0.6）是保证训练初期稳定性的核心。

## 7. 总结
- **核心思想**：利用统一编码器和零初始化融合，实现多源先验的组合与鲁棒注入。
- **速记版pipeline**：
    1. **预对齐**：使用 SE(2) 修正多源输入的位姿漂移。
    2. **独立编码**：分别对矢量和栅格进行语义与空间表征。
    3. ** gated 融合**：动态加权不同先验的置信度。
    4. **残差注入**：将先验无害化叠加至 BEV 特征。

**Key Findings:**

- We present UMPE, a Unified Map Prior Encoder that can ingest any subset of four priors and fuse them with BEV features for both mapping and planning.
- UMPE is compositional: when trained with all priors, it outperforms single-prior models even when only one prior is available at test time, demonstrating powerset robustness.
- These results show that a unified, alignment-aware treatment of heterogeneous map priors yields better mapping and better planning.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02762v1)
- [arXiv](https://arxiv.org/abs/2605.02762v1)

---

<a id='2605.02757v1'></a>
## [Seeing Realism from Simulation: Efficient Video Transfer for Vision-Language-Action Data Augmentation](https://arxiv.org/abs/2605.02757v1)

**Authors:** Chenyu Hui, Xiaodi Huang, Siyu Xu, Yunke Wang, Shan You, Fei Wang, Tao Huang, Chang Xu

**Published:** 2026-05-04

**Categories:** cs.CV, cs.RO

**Abstract:**

Vision-language-action (VLA) models typically rely on large-scale real-world videos, whereas simulated data, despite being inexpensive and highly parallelizable to collect, often suffers from a substantial visual domain gap and limited environmental diversity, resulting in weak real-world generalization. We present an efficient video augmentation framework that converts simulated VLA videos into realistic training videos while preserving task semantics and action trajectories. Our pipeline extracts structured conditions from simulation via video semantic segmentation and video captioning, rewrites captions to diversify environments, and uses a conditional video transfer model to synthesize realistic videos. To make augmentation practical at scale, we introduce a diffusion feature-reuse mechanism that reuses video tokens across adjacent timesteps to accelerate generation, and a coreset sampling strategy that identifies a compact, non-redundant subset for augmentation under limited computation. Extensive experiments on Robotwin 2.0, LIBERO, LIBERO-Plus, and a real robotic platform demonstrate consistent improvements. For example, our method improves RDT-1B by 8% on Robotwin 2.0, and boosts $π_0$ by 5.1% on the more challenging LIBERO-Plus benchmark. Code is available at: https://github.com/nanfangxiansheng/Seeing-Realism-from-Simulation.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇论文的分析如下：

### 1. 主要贡献总结
该论文提出了一种高效的视频增强框架，旨在弥合仿真数据与真实世界视觉数据之间的鸿沟，从而提升视觉-语言-动作（VLA）模型的泛化能力。通过引入结构化条件提取、环境多样化描述以及高效的条件视频迁移模型，该方法能将廉价的仿真数据转化为高质量的现实风格数据，在多个机器人基准测试中显著提升了模型性能。

### 2. 关键创新点与方法论
该工作的核心在于如何**低成本、高保真地实现跨域（Sim-to-Real）数据增强**，其技术亮点包括：
*   **语义保真的视频迁移：** 并非简单的图像风格迁移，而是通过视频语义分割和描述（Captioning）提取核心任务语义，确保生成的真实感视频在动作轨迹和任务逻辑上与原始仿真数据高度一致。
*   **扩散特征重用机制（Diffusion Feature-Reuse）：** 这是针对视频生成计算成本高昂的针对性优化。通过跨相邻时间步共享视频Token，大幅降低了计算开销，使得在大规模数据集上的增强变得现实可行。
*   **核心集采样（Coreset Sampling）：** 引入了一种非冗余的数据子集识别策略，避免了对大量相似仿真数据的重复增强，确保了计算资源的最高效利用。

### 3. 对领域的潜在影响
*   **VLA模型训练范式的变革：** 目前VLA模型极其依赖难以获取的高质量真实世界视频数据。该研究证明了利用“高质量仿真数据+高效视频迁移”替代部分真实数据的可行性，为解决机器人领域长期存在的“数据匮乏”问题提供了新路径。
*   **计算效率与生成质量的平衡：** 该工作在扩散模型推理加速方面的探索，为将生成式AI应用于大规模机器人数据增强设置了新的性能基准。

### 4. 受益的相关领域与应用
*   **具身智能（Embodied AI）：** 直接受益者是机器人决策与操作策略学习。
*   **自动驾驶：** 模拟器生成的交通场景往往存在“仿真味”，该技术可将其转化为真实的道路视觉数据，提升自动驾驶感知模型的鲁棒性。
*   **视频合成与编辑：** 其特征重用和条件控制技术对视频生成研究（如Sora类模型）的推理加速具有参考价值。

### 5. 可推断的局限性
*   **生成的一致性：** 尽管采用了特征重用，但在复杂任务中，视频的时间连续性（Temporal Consistency）和动作细节的物理合规性（Physics Plausibility）是否能完全达到真实录制视频的水平仍有待考量。
*   **语义限制：** 框架高度依赖于视频语义分割和描述模型的能力。如果原始仿真场景极其复杂或边缘案例（Edge Cases）较多，目前的提取手段可能导致信息丢失，从而限制了增强数据的“多样性上限”。
*   **泛化瓶颈：** 仿真环境本身的物理多样性限制了生成数据的边界；该方法目前似乎主要解决的是“视觉域”的鸿沟，对于复杂的物理交互动力学差异，可能仍需补充其他方法。

---
**专家点评：**
这篇论文的价值在于它将**生成式AI的技术红利**与**机器人学习的实际痛点**进行了极具工程价值的结合。它没有盲目追求纯生成模型的质量，而是通过精巧的计算优化（Token重用）和策略采样，解决了大模型训练中最让人头疼的“规模化”问题。这对于当前AI在物理世界落地过程中“数据获取难”的困境，是一次非常切中要害的尝试。

**Key Findings:**

- We present an efficient video augmentation framework that converts simulated VLA videos into realistic training videos while preserving task semantics and action trajectories.
- To make augmentation practical at scale, we introduce a diffusion feature-reuse mechanism that reuses video tokens across adjacent timesteps to accelerate generation, and a coreset sampling strategy that identifies a compact, non-redundant subset for augmentation under limited computation.
- For example, our method improves RDT-1B by 8% on Robotwin 2.0, and boosts $π_0$ by 5.1% on the more challenging LIBERO-Plus benchmark.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02757v1)
- [arXiv](https://arxiv.org/abs/2605.02757v1)

---

<a id='2605.02739v1'></a>
## [Latent Bridge: Feature Delta Prediction for Efficient Dual-System Vision-Language-Action Model Inference](https://arxiv.org/abs/2605.02739v1)

**Authors:** Yudong Liu, Yuan Li, Zijia Tang, Yuxi Zheng, Yueqian Lin, Qinsi Wang, Yi Li, Shuangjun Liu, Shuai Zhang, Taotao Jing, Dashan Gao, Ning Bi, Jingwei Sun, Yiran Chen, Hai Li

**Published:** 2026-05-04

**Categories:** cs.RO

**Abstract:**

Dual-system Vision-Language-Action (VLA) models achieve state-of-the-art robotic manipulation but are bottlenecked by the VLM backbone, which must   execute at every control step while producing temporally redundant features. We propose Latent Bridge, a lightweight model that predicts VLM output   deltas between timesteps, enabling the action head to operate on predicted outputs while the expensive VLM backbone is called only periodically. We   instantiate Latent Bridge on two architecturally distinct VLAs: GR00T-N1.6 (feature-space bridge) and π0.5 (KV-cache bridge), demonstrating that the   approach generalizes across VLA designs. Our task-agnostic DAgger training pipeline transfers across benchmarks without modification. Across four   LIBERO suites, 24 RoboCasa kitchen tasks, and the ALOHA sim transfer-cube task, Latent Bridge achieves 95-100% performance retention while reducing   VLM calls by 50-75%, yielding 1.65-1.73x net per-episode speedup.

**Analysis:**

### 1. 摘要翻译
双系统视觉-语言-动作（VLA）模型在机器人操作任务中表现优异，但受限于VLM主干网络。VLM必须在每个控制步骤执行，且产生了大量的时序冗余特征。本文提出了“Latent Bridge”，这是一个轻量级模型，能够预测时间步之间的VLM输出增量（delta）。这使得动作头可以在预测出的输出上运行，而昂贵的VLM主干只需周期性调用。我们在两种架构迥异的VLA（GR00T-N1.6和π0.5）上进行了实例化，证明了该方法对不同VLA架构的通用性。我们在四个LIBERO套件、24个RoboCasa厨房任务以及ALOHA模拟任务上进行了验证，结果显示Latent Bridge在减少50–75% VLM调用次数的同时，实现了95–100%的性能保持，并获得了1.65–1.73倍的实际运行速度提升。

### 2. 方法动机分析
*   **驱动力**：解决双系统VLA中VLM主干计算密集、推理延迟高且在时间维度上存在冗余特征的问题。
*   **痛点**：现有加速方法（如token剪枝）在双系统架构中面临“Amdahl瓶颈”——仅加速VLM主干部分，忽略了动作头推理耗时，导致整体提速有限。
*   **核心假设**：连续时间步的VLM输出具有高度的时间相关性（ temporal redundancy），因此可以通过一个极小的模型预测特征增量（$\Delta_t = z_{t+1} - z_t$），从而跳过中间步骤的VLM计算。

### 3. 方法设计详解
*   **核心 pipeline**：
    1.  **Sync Data Collection**：在模拟器中运行全量VLM，记录轨迹数据。
    2.  **R0 Bridge Training**：训练轻量化DiT（Diffusion Transformer）作为增量预测器，使用MSE和余弦相似度损失函数，仅对图像token进行预测。
    3.  **DAgger Refinement (R1)**：通过在线策略，让桥模型在推理时自我修正，闭环纠正预测误差。
*   **模型结构**：
    *   **GR00T版**：特征空间桥，基于DiT架构，通过交叉注意力（cross-attention）获取稳定场景上下文。
    *   **π0.5版**：KV缓存桥，对18层Gemma层的KV进行并行预测。
*   **关键公式**：$\hat{z}_{t+1} = \hat{z}_t + \mathcal{B}(\hat{z}_t, s_t, q_t, a_{t-1})$。其中$\hat{z}$是预测特征，$s_t$是缓存的稳定中间特征，$q$和$a$分别为机器人状态和动作。

### 4. 方法对比分析
*   **本质区别**：与现有优化方法（如FastV、VLA-Cache）仅通过剪枝或局部更新不同，Latent Bridge在中间步骤**完全跳过**了VLM的Forward过程。
*   **创新贡献**：提出了一种通用的、任务不可知（task-agnostic）的桥接预测架构；将预测目标从图像token扩展到KV缓存；引入DAgger训练方案解决自回归预测带来的漂移问题。
*   **适用场景**：适用于所有采用双系统架构（VLM Backbone + Action Head）的机器人策略模型。

### 5. 实验分析
*   **验证方法**：在多个大规模机器人任务基准上，对比了Sync（基准线）、FastV、VLA-Cache及本文方法。
*   **核心结论**：在保持性能几乎无损（95-100%）的前提下，实现了约1.7倍的实际墙钟时间加速，显著优于传统的剪枝方案。
*   **优势**：极低的桥接计算延迟（2-6ms），高性能保留度。
*   **局限**：对模型Checkpoint有依赖（需针对性微调）；在线DAgger阶段需要模拟器支持。

### 6. 实用指南
*   **开源信息**：已开源，代码库见论文脚注（https://github.com/1999Lyd/Latent-Bridge）。
*   **训练建议**：
    *   必须使用`torch.compile`和`bf16`以达到实验描述的加速比。
    *   DAgger阶段的学习率需设置为R0阶段的1/10，以稳定分布偏移。
*   **迁移步骤**：针对新VLA模型，仅需运行Sync数据采集，训练一个轻量级的DiT进行增量预测即可。

### 7. 总结
*   **核心思想**：通过预测特征时序增量，在双系统架构中间步跳过VLM推理。
*   **速记版Pipeline**：
    1. 收集全量推理数据；
    2. 训练轻量化增量预测器；
    3. 通过DAgger策略消除模型误差累积；
    4. 部署时周期性运行VLM，其余步由预测器补齐。

**Key Findings:**

- Dual-system Vision-Language-Action (VLA) models achieve state-of-the-art robotic manipulation but are bottlenecked by the VLM backbone, which must   execute at every control step while producing temporally redundant features.
- We propose Latent Bridge, a lightweight model that predicts VLM output   deltas between timesteps, enabling the action head to operate on predicted outputs while the expensive VLM backbone is called only periodically.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02739v1)
- [arXiv](https://arxiv.org/abs/2605.02739v1)

---

<a id='2605.02730v1'></a>
## [Perceptual Flow Network for Visually Grounded Reasoning](https://arxiv.org/abs/2605.02730v1)

**Authors:** Yangfu Li, Yuning Gong, Hongjian Zhan, Teng Li, Yuanhuiyi Lyu, Tianyi Chen, Qi Liu, Ziyuan Huang, Zhihang Zhong, Dandan Zheng, Yue Lu

**Published:** 2026-05-04

**Categories:** cs.CV, cs.AI

**Abstract:**

Despite the success of Large-Vision Language Models (LVLMs), general optimization objectives (e.g., standard MLE) fail to constrain visual trajectories, leading to language bias and hallucination. To mitigate this, current methods introduce geometric priors from visual experts as additional supervision. However, we observe that such supervision is typically suboptimal: it is biased toward geometric precision and offers limited reasoning utility. To bridge this gap, we propose Perceptual Flow Network (PFlowNet), which eschews rigid alignment with the expert priors and achieves interpretable yet more effective visual reasoning. Specifically, PFlowNet decouples perception from reasoning to establish a self-conditioned generation process. Based on this, it integrates multi-dimensional rewards with vicinal geometric shaping via variational reinforcement learning, thereby facilitating reasoning-oriented perceptual behaviors while preserving visual reliability. PFlowNet delivers a provable performance guarantee and competitive empirical results, particularly setting new SOTA records on V* Bench (90.6%) and MME-RealWorld-lite (67.0%).

**Analysis:**

### 1. 摘要翻译
尽管大视觉语言模型（LVLMs）取得了成功，但标准的最大似然估计（MLE）等通用优化目标无法有效约束视觉轨迹，导致语言偏见和幻觉。为缓解此问题，现有方法引入视觉专家的几何先验作为监督，但这种监督通常次优，且偏向几何精度而缺乏推理效用。为此，本文提出了**感知流网络（PFlowNet）**。PFlowNet摒弃了与专家先验的刚性对齐，解耦了感知与推理，通过变分强化学习将多维度奖励与近邻几何整形（vicinal geometric shaping）相结合，在建立自条件生成过程的同时，实现了以推理为导向的感知行为，并保持了视觉可靠性。实验表明，PFlowNet在V* Bench（90.6%）和MME-RealWorld-lite（67.0%）上设定了新的SOTA。

### 2. 方法动机分析
- **痛点**：现有基于RLVR的方法往往通过最大化视觉预测与预训练专家检测器（如GroundingDINO）的几何一致性来约束推理。然而，这种“刚性对齐”不仅忽视了推理效用（产生“隧道效应”），还引入了专家先验的偏见，导致模型难以跳出专家定义的框框进行更全面的视觉理解。
- **研究假设**：视觉推理的关键不在于追求几何的绝对精确，而在于通过自参数化的变分过程，寻找对解决具体问题最有帮助的感知流（Perceptual Flow），即“视觉想法”。

### 3. 方法设计详解
- **核心组件：感知流 (Perceptual Flow)**
  - 结构：$Z = (z_0 \to z_1 \dots z_K)$，包含一个规划状态（$z_0$，分析查询）和多个感知状态（$z_{\ge 1}$，即RoI区域与描述）。
- **流程pipeline**：
  1. **冷启动训练**：利用教师模型（如Gemini/GPT-4o）合成细粒度 trajectories，通过监督微调（SFT）引导模型生成感知流。
  2. **变分强化微调（RFT）**：
     - **解耦推理**：模型先采样感知流 $Z$，再基于流进行推理 $p_\theta(Y|Z, X)$。
     - **多维奖励**：联合优化质量（与专家定义的视觉上下文的一致性）和效用（对最终回答 $Y$ 的信息增益）。
     - **近邻几何整形**：引入 $\omega_\lambda(z_{0:k}, E)$，只对那些“严重偏离”专家先验的轨迹进行惩罚，而非强制完全对齐，从而在保证合理性和鼓励探索之间取得平衡。
- **目标函数**：采用分层变分目标 **Sub-Trajectory Balance (SubTB)**，提供稠密的中间监督，优化 $p_\theta(Z|X)$。

### 4. 方法对比分析
- **本质区别**：从“模仿专家”转变为“以推理效用为导向的自主探索”。
- **创新贡献**：提出感知流表征，通过变分强化学习实现感知与推理的解耦，并设计了近邻几何整形机制以规避专家偏见。
- **适用场景**：极度依赖细粒度视觉证据（如计数、空间关系、图表阅读）的复杂场景。

### 5. 实验分析
- **结果**：在V* Bench上达到90.6%，TreeBench和MME-RealWorld-lite上显著超越Base Model。
- **优势**：性能-效率平衡优异（相比Agent框架更短的Context），推理过程具备高解释性。
- **局限**：缺乏自适应感知（对所有问题采用固定推理流程），处理STEM类简单问题时可能引入不必要的开销。

### 6. 实用指南
- **迁移可能**：该架构可以作为一种通用的推理范式，迁移至任何需要多步骤视觉搜索的VLM。
- **实现关键**：
  - `SubTB`公式的实现是稳定训练的核心。
  - 数据预处理中的“随机扩展”是缓解专家偏见的关键步骤。
  - 需要一个预训练好的Reward Model $p_\phi$ 来计算对比奖励。

### 7. 总结
- **核心思想**：通过解耦感知与推理，变分强化学习实现对视觉注意力的“推理导向”优化。
- **速记pipeline**：
  1. **规划**：分析问题，生成思考路径。
  2. **感知**：根据思考，按需对感兴趣区域（RoI）进行采样。
  3. **推理**：基于采样的视觉流及Zoom-in细节，输出结论。

**Key Findings:**

- To bridge this gap, we propose Perceptual Flow Network (PFlowNet), which eschews rigid alignment with the expert priors and achieves interpretable yet more effective visual reasoning.
- PFlowNet delivers a provable performance guarantee and competitive empirical results, particularly setting new SOTA records on V* Bench (90.6%) and MME-RealWorld-lite (67.0%).

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02730v1)
- [arXiv](https://arxiv.org/abs/2605.02730v1)

---

<a id='2605.02667v1'></a>
## [AnchorD: Metric Grounding of Monocular Depth Using Factor Graphs](https://arxiv.org/abs/2605.02667v1)

**Authors:** Simon Dorer, Martin Büchner, Nick Heppert, Abhinav Valada

**Published:** 2026-05-04

**Categories:** cs.RO, cs.CV

**Abstract:**

Dense and accurate depth estimation is essential for robotic manipulation, grasping, and navigation, yet currently available depth sensors are prone to errors on transparent, specular, and general non-Lambertian surfaces. To mitigate these errors, large-scale monocular depth estimation approaches provide strong structural priors, but their predictions can be potentially skewed or mis-scaled in metric units, limiting their direct use in robotics. Thus, in this work, we propose a training-free depth grounding framework that anchors monocular depth estimation priors from a depth foundation model in raw sensor depth through factor graph optimization. Our method performs a patch-wise affine alignment, locally grounding monocular predictions in metric real-world depth while preserving fine-grained geometric structure and discontinuities. To facilitate evaluation in challenging real-world conditions, we introduce a benchmark dataset with dense scene-wide ground truth depth in the presence of non-Lambertian objects. Ground truth is obtained via matte reflection spray and multi-camera fusion, overcoming the reliance on object-only CAD-based annotations used in prior datasets. Extensive evaluations across diverse sensors and domains demonstrate consistent improvements in depth performance without any (re-)training. We make our implementation publicly available at https://anchord.cs.uni-freiburg.de.

**Analysis:**

### 1. 摘要翻译
密集且准确的深度估计对机器人操作、抓取和导航至关重要，但目前的深度传感器在处理透明、镜面反射等非朗伯体（non-Lambertian）表面时极易出错。为缓解这一问题，现有的大规模单目深度估计（MDE）方法虽能提供较强的结构先验，但其预测结果往往存在尺度偏移或偏差，限制了其在机器人领域的直接应用。为此，我们提出了一种无需训练的深度接地（grounding）框架——AnchorD。该方法通过因子图优化，将深度基础模型的单目先验锚定到原始传感器深度中。我们的方法通过块级仿射对齐（patch-wise affine alignment），在保持精细几何结构和不连续性的同时，将单目预测局部接地于真实物理空间。此外，为促进在复杂现实场景下的评估，我们引入了一个包含密集场景级真实深度信息的基准数据集，通过哑光喷涂和多相机融合技术克服了对CAD模型标注的依赖。实验证明，该方法无需任何（再）训练即可在多种传感器和场景下显著提升深度估计精度。项目开源地址：https://anchord.cs.uni-freiburg.de。

### 2. 方法动机分析
*   **驱动力**：解决机器人感知中，传统RGB-D传感器在非朗伯体表面（如透明、反光物体）深度失效的痛点，同时纠正通用单目深度估计模型（MDE）预测结果的尺度不确定性。
*   **痛点**：现有方法通常依赖监督学习，需要昂贵的标注数据，且对环境变化敏感（跨域性能差）。此外，单纯的全局仿射缩放无法纠正局部畸变，难以满足机器人精细操作的几何精度需求。
*   **核心假设**：虽然传感器在非朗伯表面数据丢失，但RGB图像保留了丰富的边缘、语义和几何结构。通过将MDE的“结构先验”与稀疏的“传感器观测”在局部仿射空间内进行因子图联合优化，可以恢复出精细、可信的度量深度。

### 3. 方法设计详解
*   **流程总结**：
    1.  **MDE预处理**：使用现成的单目深度基础模型（如DepthAnything3）生成初始稠密深度图。
    2.  **块级划分**：将场景划分为固定大小的 $m \times m$ 区域，每一块赋予独立的仿射参数 $\{s_i, b_i\}$。
    3.  **因子图构建**：构建包含三类因子的目标函数：
        *   **MDE对齐因子 ($\phi^{\text{mde}}$)**：强制优化后的深度符合单目预测的结构分布。
        *   **传感器一致性因子 ($\phi^{\text{sen}}$)**：将深度锚定在可靠的传感器测量值上。
        *   **斜率一致性因子 ($\phi^{\text{slp}}$)**：正则化相邻像素的相对深度变化，跨块边界保持平滑。
    4.  **迭代优化**：采用IRLS（迭代重加权最小二乘）求解最优深度 $D^*$ 及参数 $s, b$。
    5.  **高斯平滑**：利用高斯加权平均整合局部块仿射参数，消除块边界效应，生成最终稠密深度。

### 4. 方法对比分析
*   **本质区别**：与现有监督学习方法相比，AnchorD 是一个**无需训练（training-free）的优化框架**，即插即用，不依赖特定数据集的拟合，具有极强的跨域泛化能力。
*   **创新贡献**：提出“块级仿射接地”方案，在维持全局一致性的前提下，赋予模型局部纠正能力的灵活性。
*   **适用场景**：机器人实时抓取、桌面物体操作、需要高质量稠密深度地图且传感器受环境光照干扰的场景。

### 5. 实验分析
*   **验证方法**：在自建的SprayD数据集（含真实环境地面真值）和ClearPose数据集上进行评估，对比多种基线（Inpainting、全局仿射对齐、基础MDE模型）。
*   **关键结论**：在所有指标（MAE, RMSE, REL）上表现均优于现有方法，特别是在处理透明、遮挡等困难物体区域。
*   **优势/局限**：优势在于泛化性强、对结构保持优秀；局限是优化过程增加了计算耗时（尽管通过GPU并行加速），且依赖单目模型提供的初始结构先验。

### 6. 实用指南
*   **开源情况**：已开源 (https://anchord.cs.uni-freiburg.de)。
*   **实现细节**：建议使用 `jaxLS` 库以充分发挥GPU并行求解因子图的效率。需特别注意超参数 $\lambda^{\text{mde}}=2.5, \lambda^{\text{sen}}=0.5$ 的权衡，这是平衡“先验约束”与“传感器观测”的关键。
*   **迁移建议**：该方法模块化设计，可以轻松更换为任何更新的单目深度基础模型作为先验输入，无需任何架构改动。

### 7. 总结
*   **核心思想**：通过因子图局部仿射对齐，将单目深度接地为高精度度量深度。
*   **速记版pipeline**：
    1. 输入RGB生成初始单目深度；
    2. 划分为多个局部方块；
    3. 联合优化深度值与局部仿射缩放参数；
    4. 高斯加权平滑块边界。

**Key Findings:**

- Thus, in this work, we propose a training-free depth grounding framework that anchors monocular depth estimation priors from a depth foundation model in raw sensor depth through factor graph optimization.
- Our method performs a patch-wise affine alignment, locally grounding monocular predictions in metric real-world depth while preserving fine-grained geometric structure and discontinuities.
- To facilitate evaluation in challenging real-world conditions, we introduce a benchmark dataset with dense scene-wide ground truth depth in the presence of non-Lambertian objects.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02667v1)
- [arXiv](https://arxiv.org/abs/2605.02667v1)

---

<a id='2605.02641v1'></a>
## [Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE](https://arxiv.org/abs/2605.02641v1)

**Authors:** Yangming Shi, Shixiang Zhu, Tao Shen, Zhimiao Yu, Dengsheng Chen, Taicai Chen, Yunfei Yang, Juan Zhou, Chen Cheng, Liang Ma, Xibin Wu, Benxuan Yan, Ge Li, Tuoyu Zhang, Dan Li, Chang Liu, Zhenbang Sun

**Published:** 2026-05-04

**Categories:** cs.CV

**Abstract:**

We present Mamoda2.5, a unified AR-Diffusion framework that seamlessly integrates multimodal understanding and generation within a single architecture. To efficiently enhance the model's generation capability, we equip the Diffusion Transformer backbone with a fine-grained Mixture-of-Experts (MoE) design (128 experts, Top-8 routing), yielding a 25B-parameter model that activates only 3B parameters, significantly reducing training costs while scaling up the model capacity. Mamoda2.5 achieves top-tier generation performance on VBench 2.0 and sets a new record in video editing quality, surpassing evaluated open-source models and matching the performance of current top-tier proprietary models, including the Kling O1 on OpenVE-Bench. Furthermore, we introduce a joint few-step distillation and reinforcement learning framework that compresses the 30-step editing model into a 4-step model and greatly accelerates model inference. Compared to open-source baselines, Mamoda2.5 achieves up to $95.9\times$ faster video editing inference. In real-world applications, Mamoda2.5 has been successfully deployed for content moderation and creative restoration tasks in advertising scenarios, achieving a 98% success rate in internal advertising video editing scenario.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对 **Mamoda2.5** 的分析如下：

### 1. 核心贡献总结
Mamoda2.5 提出了一种创新的统一架构，将多模态理解与生成整合于单一框架内。该研究通过引入 **DiT-MoE (Diffusion Transformer + Mixture-of-Experts)** 架构，在维持 25B 参数容量的同时，通过稀疏激活极大地提升了推理效率；同时，通过联合蒸馏与强化学习实现了视频编辑的高效加速，在性能与实用性上达到了业界领先水平。

### 2. 关键创新与方法论
*   **架构范式升级 (DiT-MoE):** 将传统的 Diffusion Transformer 与细粒度 MoE (128 个专家，Top-8 路由) 结合，仅需激活 3B 参数。这种设计完美解决了大参数规模带来的计算成本瓶颈，实现了“高性能与轻量化”的平衡。
*   **高效推理策略:** 提出“几步蒸馏 + 强化学习”的联合优化路径，将生成/编辑任务从传统的 30 步压缩至 4 步。这种从模型架构到采样策略的全栈优化，实现了 95.9 倍的推理加速。
*   **统一任务框架:** 实现了理解（Understanding）与生成（Generation）的架构统一，这反映了当前 CV 领域向“多模态基础模型（Foundation Model）”迈进的核心趋势。

### 3. 对领域的潜在影响
*   **MoE 在生成模型中的规模化应用:** 该论文证明了 MoE 架构不仅适用于大语言模型，同样能极大地释放视觉生成模型（特别是 DiT）的潜力，为后续构建超大规模生成模型提供了可行路径。
*   **工业化落地的新标杆:** 相比于仅关注学术指标的工作，Mamoda2.5 在广告审核与创意修复场景中的 98% 成功率证明了生成式 AI 跨越“技术原型”进入“生产环境”的潜力，为商业化闭环提供了范式。
*   **基准测试的重构:** 其在 VBench 2.0 和 OpenVE-Bench 上的优异表现，标志着开源模型在视频生成领域正在迅速缩小与闭源（如 Kling O1）的差距。

### 4. 受益的相关领域与应用
*   **视频编辑与创作:** 极速的推理速度使得端侧或实时视频特效处理成为可能。
*   **内容治理与合规:** 论文提到的广告审核场景证明了其在自动化视频语义审查和内容修复方面的强大价值。
*   **广告与媒体工业:** 自动化的创意生成、智能修补与重定向，能够显著降低营销内容的制作成本。

### 5. 可推断的局限性
*   **训练复杂性:** 尽管推理时节省了算力，但 MoE 架构的训练（如专家负载均衡、路由稳定性）通常面临收敛难度大、训练不稳定的问题。
*   **蒸馏带来的泛化损失:** 4-step 蒸馏虽然大幅提升了速度，但通常会伴随生成多样性或微观细节质量的细微下降，可能在极端复杂的语义生成中表现不如全量模型。
*   **视觉一致性挑战:** 作为统一模型，如何在“理解”与“生成”两个任务之间平衡权重（防止生成任务干扰理解的准确性，或理解任务制约生成的创造性），是此类架构长期面临的技术博弈。

**总结：** Mamoda2.5 是目前视频生成领域从“堆参数”转向“高效架构设计”的代表作，其将稀疏激活与蒸馏加速结合的思路，为构建大规模、可落地的视频生成基础模型提供了极具参考价值的路径。

**Key Findings:**

- We present Mamoda2.5, a unified AR-Diffusion framework that seamlessly integrates multimodal understanding and generation within a single architecture.
- Mamoda2.5 achieves top-tier generation performance on VBench 2.0 and sets a new record in video editing quality, surpassing evaluated open-source models and matching the performance of current top-tier proprietary models, including the Kling O1 on OpenVE-Bench.
- Furthermore, we introduce a joint few-step distillation and reinforcement learning framework that compresses the 30-step editing model into a 4-step model and greatly accelerates model inference.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02641v1)
- [arXiv](https://arxiv.org/abs/2605.02641v1)

---

<a id='2605.02638v1'></a>
## [ViewSAM: Learning View-aware Cross-modal Semantics for Weakly Supervised Cross-view Referring Multi-Object Tracking](https://arxiv.org/abs/2605.02638v1)

**Authors:** Jiawei Ge, Xintian Zhang, Jiuxin Cao, Bo Liu, Fabian Deuser, Chang Liu, Gong Wenkang, Siyou Li, Juexi Shao, Wenqing Wu, Chen Feng, Ioannis Patras

**Published:** 2026-05-04

**Categories:** cs.CV, cs.AI

**Abstract:**

Cross-view Referring Multi-Object Tracking (CRMOT) aims to track multiple objects specified by natural language across multiple camera views, with globally consistent identities. Despite recent progress, existing methods rely heavily on costly frame-level spatial annotations and cross-view identity supervision. To reduce such reliance, we explore CRMOT under weak supervision by leveraging the capabilities of foundation models. However, our empirical study shows that directly applying foundation models such as SAM2 and SAM3, even with task-specific modifications, fails to accurately understand referring expressions and maintain consistent identities across views. Yet, they remain effective at producing reliable object tracklets that can serve as pseudo supervision. We therefore repurpose foundation models as pseudo-label generators and propose a two-stage framework for weakly supervised CRMOT, using only object category labels as coarse-grained supervision. In the first stage, we design an Affinity-guided Cross-view Re-prompting strategy to refine and associate SAM3-generated tracklets across cameras, producing reliable cross-view pseudo labels for subsequent training. In the second stage, we introduce ViewSAM, a CRMOT model built upon SAM2 that explicitly models view-aware cross-modal semantics. By formulating view-induced variations as learnable conditions, ViewSAM bridges the gap between view-variant visual observations and view-invariant textual expressions, enabling robust cross-view referring tracking with only approximately 10% additional parameters. Extensive experiments demonstrate that ViewSAM achieves SOTA performance under weak supervision and remains competitive with fully supervised methods.

**Analysis:**

以下是对《ViewSAM: Learning View-aware Cross-modal Semantics for Weakly Supervised Cross-view Referring Multi-Object Tracking》的深度分析：

### 1. 摘要翻译
跨视角指称多目标跟踪（CRMOT）旨在根据自然语言描述，在多个摄像机视角下跟踪全局身份一致的目标。现有的方法严重依赖高昂的帧级空间标注和跨视角身份监督。为降低这种依赖，我们探索了弱监督下的CRMOT，利用基础模型（Foundation Models）的能力。实证研究表明，直接应用SAM2和SAM3等模型，即使经过特定修改，也无法准确理解指称表达并保持视角间的身份一致性，但它们在生成可靠的目标轨迹片段（tracklets）方面依然有效。因此，我们将基础模型重新定位为伪标签生成器，提出了一个两阶段的弱监督CRMOT框架，仅使用对象类别标签作为粗粒度监督。第一阶段，我们设计了“亲和力引导的跨视角重新提示”（Affinity-guided Cross-view Re-prompting）策略，以细化和关联跨摄像头的轨迹片段，产生可靠的伪标签。第二阶段，我们引入了ViewSAM，一个构建在SAM2之上的CRMOT模型，能够明确建模视角感知的跨模态语义。通过将视角引起的差异视为可学习的条件，ViewSAM弥合了视角变化的视觉观测与视角不变的文本表达之间的差距，仅需约10%的额外参数即可实现鲁棒的跨视角指称跟踪。实验表明，ViewSAM在弱监督下达到了SOTA性能，并与全监督方法具有竞争力。

### 2. 方法动机分析
*   **驱动力**：旨在彻底摆脱CRMOT对人工密集型标注（空间坐标、跨视角身份）的依赖，实现基于弱监督（仅类别标签）的规模化应用。
*   **痛点**：基础模型（如SAM2/3）在处理长时跟踪时存在“跟踪偏差”（tracking bias），易漂移到视觉相似的干扰项；且无法处理不同视角下目标外观和运动模式的显著差异，难以维持全局ID一致性。
*   **核心假设**：尽管基础模型直接用于CRMOT效果不佳，但其拥有极强的单视图跟踪底座能力，可作为“伪标签生成器”；此外，若将视角变异视为一种“条件”而非干扰，通过跨模态语义对齐，即可实现鲁棒的跨视角指称跟踪。

### 3. 方法设计详解
#### 流程总结
1.  **阶段一：伪标签生成**
    *   **SAM3跟踪**：利用类别标签提示SAM3获取各摄像头的单视图轨迹片段。
    *   **亲和力关联**：计算轨迹片段的特征亲和力，进行初步跨视角关联。
    *   **双向重新提示（Bi-directional Re-prompting）**：建立身份原型（Identity Prototype），通过LLM生成描述，对轨迹进行迭代更新和细化。
2.  **阶段二：ViewSAM模型学习**
    *   **VC-CMA模块**：引入可学习的“动态视角Token”，通过Cross-Attention将视角先验注入视觉与文本特征，实现视角感知的跨模态对齐。
    *   **偏见校准（BAR）**：通过比较“记忆增强”与“无记忆”预测，自动检测漂移，若漂移则向无记忆预测回退。
    *   **CGCT跟踪头**：引入“一致性引导”的特征空间，强制执行视角内的时间一致性及跨视角的全局身份对齐。

### 4. 方法对比分析
*   **本质区别**：不试图去噪或消除视角变异，而是主动建模视角变异（View-aware），将视角变异作为训练的辅助条件。
*   **创新贡献**：提出将基础模型与弱监督学习解耦为“生成器”与“强化器”；提出动态视角Token与偏见校准模块，有效解决了跨视角下的身份一致性难题。

### 5. 实验分析
*   **有效性**：在CRTrack基准测试中，ViewSAM在弱监督设置下大幅领先现有SOTA，性能甚至逼近全监督基线。
*   **局限性**：高度依赖阶段一生成的伪标签质量；系统复杂度高，需运行多个外部模型（SAM3、MLLM、ReID）。

### 6. 实用指南
*   **实现细节**：核心训练包括对VC-CMA、Candidate Generator、BAR和CGCT的轻量化微调（总参数量增加约10%），SAM2主干保持冻结。
*   **迁移建议**：该“先生成伪标签，再训练小模块适配”的框架，可直接迁移至工业场景下的多摄像头监控、机器人视觉感知等弱标注视频分析任务。

### 7. 总结
*   **核心思想**：通过建模视角变异作为可学习条件，实现弱监督下的鲁棒跨视角跟踪。
*   **速记版Pipeline**：
    1. 用SAM3追踪目标获取初步轨迹；
    2. 利用亲和力和LLM描述生成高质量跨视角伪标签；
    3. 用动态视角Token注入，使模型“感知”相机视角；
    4. 增加偏见校准，修正模型漂移；
    5. 通过一致性约束，实现跨视角身份统一。

**Key Findings:**

- In the second stage, we introduce ViewSAM, a CRMOT model built upon SAM2 that explicitly models view-aware cross-modal semantics.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.02638v1)
- [arXiv](https://arxiv.org/abs/2605.02638v1)

---

