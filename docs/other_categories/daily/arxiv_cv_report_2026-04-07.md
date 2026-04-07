time: 20260407

# Arxiv Computer Vision Papers - 2026-04-07

## Executive Summary

### **Arxiv 计算机视觉领域论文日报执行摘要 (2026-04-06)**

**1. 核心主题与趋势概览**

今日的论文集体现了计算机视觉领域几个强劲的交叉融合趋势：

*   **3D场景理解的效率与泛化性**：多篇论文聚焦于提升3D场景重建、理解和编辑的效率与质量。核心方向包括**动态网络参数调整**（PointTPA）、**非网格对齐的3D高斯表示**（Free-Range Gaussians）以及利用**纯规则生成合成数据**（Fully Procedural Synthetic Data）来提升多视图立体的泛化能力。
*   **生成式模型的新兴应用**：研究正深入挖掘预训练生成模型（尤其是扩散模型）的潜力，将其能力拓展至**图像修复**（Your Pre-trained Diffusion Model...）和**高效世界建模**（A Frame is Worth One Token...），展示了“一模型多用”的潜力。
*   **具身智能与机器人交互**：将视觉感知与具体行动和用户意图相结合的研究日益突出。今日论文涵盖了**事件相机增强的视觉-语言-动作模型**（E-VLA）以应对恶劣视觉环境，以及**将草图直接翻译为机器人指令**（AnyUser）的直观人机交互。
*   **模型鲁棒性与适应性分析**：在追求性能的同时，对模型在**特定领域（如自动驾驶）微调后产生的“遗忘”问题**（The Blind Spot of Adaptation）进行量化与缓解，成为确保模型安全可靠部署的关键研究方向。

**2. 重点与创新性论文亮点**

*   **《Your Pre-trained Diffusion Model Secretly Knows Restoration》**：**极具启发性的工作**。它揭示了大规模预训练的扩散模型内部已隐含着强大的图像修复先验知识，无需专门训练或仅需极简调整即可实现高质量修复。这挑战了传统任务专用模型的范式，为挖掘基础模型的“隐性知识”开辟了新路径。
*   **《Free-Range Gaussians: Non-Grid-Aligned Generative 3D Gaussian Reconstruction》**：在火爆的3D高斯泼溅（3DGS）领域提出了重要创新。通过**解除高斯单元与规则网格的绑定**，实现了更灵活、高质量且内存高效的3D生成与重建，是3D表示学习的一个值得关注的发展。
*   **《The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models》**：**具有重要实践意义的研究**。它系统性地量化了自动驾驶模型在针对新场景微调时发生的灾难性遗忘问题，并提出了缓解策略。这对于任何考虑将大模型部署到安全关键领域的研究者和工程师都至关重要。

**3. 新兴研究方向与技术**

*   **“挖掘”而非“重建”基础模型能力**：从预训练模型中提取其未经明确训练但已具备的能力（如修复、编辑），成为一个低成本的性能提升策略。
*   **面向开放世界的具身交互**：研究重点从封闭环境转向如何让机器理解并执行**开放、模糊的用户意图**（如草图），并与**动态、非理想的真实物理环境**（黑暗、模糊）进行交互。
*   **动态与自适应的网络结构**：根据输入数据（如3D点云）动态调整网络参数（PointTPA），以提高计算效率和任务性能，是模型设计的一个前沿方向。
*   **合成数据生成的“轻量化”**：探索仅通过简单规则和程序化方法生成高质量、可用于训练复杂模型（如MVS）的合成数据，以降低数据获取成本。

**4. 推荐精读论文**

根据研究方向的普适性和影响力，建议优先阅读以下论文：

1.  **《Your Pre-trained Diffusion Model Secretly Knows Restoration》**：**必读**。其核心思想（挖掘基础模型隐性知识）可能适用于计算机视觉乃至AI的众多子领域，具有很高的启发价值。
2.  **《The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models》**：**推荐给所有从事模型适配、领域自适应及安全关键应用的研究者**。它提出了一个实际部署中无法回避的根本性问题。
3.  **《Free-Range Gaussians: Non-Grid-Aligned Generative 3D Gaussian Reconstruction》**：**推荐给从事3D视觉、神经渲染和生成模型的研究者**。代表了3D表示学习的一个最新技术进展。
4.  **《E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes》**：**推荐给研究具身智能、多模态融合和事件相机的团队**。它展示了如何通过新颖的传感器和模型架构解决传统视觉的瓶颈问题。

**总结**：今日的论文集合反映了计算机视觉领域正朝着**更高效、更通用、更贴近真实物理世界交互**的方向快速发展。研究前沿在于如何巧妙利用已有大模型、设计更灵活的表示方法，并确保模型在动态环境中的稳健性与安全性。

---

## Table of Contents

1. [PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding](#2604.04933v1)
2. [LoMa: Local Feature Matching Revisited](#2604.04931v1)
3. [Fully Procedural Synthetic Data from Simple Rules for Multi-View Stereo](#2604.04925v1)
4. [Your Pre-trained Diffusion Model Secretly Knows Restoration](#2604.04924v1)
5. [A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](#2604.04913v1)
6. [HorizonWeaver: Generalizable Multi-Level Semantic Editing for Driving Scenes](#2604.04887v1)
7. [Free-Range Gaussians: Non-Grid-Aligned Generative 3D Gaussian Reconstruction](#2604.04874v1)
8. [The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](#2604.04857v1)
9. [E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes](#2604.04834v1)
10. [AnyUser: Translating Sketched User Intent into Domestic Robots](#2604.04811v1)

---

## Papers

<a id='2604.04933v1'></a>
## [PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding](https://arxiv.org/abs/2604.04933v1)

**Authors:** Siyuan Liu, Chaoqun Zheng, Xin Zhou, Tianrui Feng, Dingkang Liang, Xiang Bai

**Published:** 2026-04-06

**Categories:** cs.CV

**Abstract:**

Scene-level point cloud understanding remains challenging due to diverse geometries, imbalanced category distributions, and highly varied spatial layouts. Existing methods improve object-level performance but rely on static network parameters during inference, limiting their adaptability to dynamic scene data. We propose PointTPA, a Test-time Parameter Adaptation framework that generates input-aware network parameters for scene-level point clouds. PointTPA adopts a Serialization-based Neighborhood Grouping (SNG) to form locally coherent patches and a Dynamic Parameter Projector (DPP) to produce patch-wise adaptive weights, enabling the backbone to adjust its behavior according to scene-specific variations while maintaining a low parameter overhead. Integrated into the PTv3 structure, PointTPA demonstrates strong parameter efficiency by introducing two lightweight modules of less than 2% of the backbone's parameters. Despite this minimal parameter overhead, PointTPA achieves 78.4% mIoU on ScanNet validation, surpassing existing parameter-efficient fine-tuning (PEFT) methods across multiple benchmarks, highlighting the efficacy of our test-time dynamic network parameter adaptation mechanism in enhancing 3D scene understanding. The code is available at https://github.com/H-EmbodVis/PointTPA.

**Analysis:**

这是一份针对 **PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding** 的深度论文分析。

### 1. 摘要翻译
场景级点云理解因几何多样性、类别分布极度不平衡及空间布局多变而充满挑战。现有方法虽提升了对象级性能，但因推理时依赖静态网络参数，限制了对动态场景数据的适应性。为此，我们提出了 **PointTPA**，这是一种针对场景级点云的测试时参数自适应框架。PointTPA 采用基于序列化的邻域分组（SNG）形成局部相干补丁，并通过动态参数投影器（DPP）生成补丁级的自适应权重，使主干网络能在保持极低参数开销的前提下，根据场景特定变化调整自身行为。集成到 PTv3 结构中，PointTPA 在 ScanNet 验证集上达到了 78.4% 的 mIoU，在保持参数高效性的同时，超越了现有的参数高效微调（PEFT）方法。

### 2. 方法动机分析
*   **驱动力**：旨在解决预训练模型在迁移到复杂、大规模 3D 场景时，因缺乏针对特定场景的动态适应能力而导致的性能瓶颈。
*   **痛点**：现有 PEFT 方法主要针对对象级任务，忽略了场景级点云存在的高度几何变异、长尾分布及复杂空间结构，无法在推理阶段提供“输入驱动”的动态参数调整。
*   **研究假设**：通过在推理时根据局部输入特征实时调整网络投影权重，能够显著提升模型对多样化场景的泛化能力与理解深度。

### 3. 方法设计详解
PointTPA 的核心在于**“输入感知”的动态适应**，流程如下：
*   **SNG（序列化邻域分组）**：将无序的 3D 点云通过空间填充曲线（如 Hilbert 或 Z-order）映射为有序序列，再划分为多个局部补丁。此过程保持了局部几何的连续性，为后续动态适配奠定结构基础。
*   **DPP（动态参数投影器）**：
    *   **参数基集（Parameter Base Set）**：预设一组可学习的权值基，作为动态空间的基础。
    *   **权重路由（Weight Router）**：通过平均池化提取补丁特征，经 MLP 和 Softmax 生成针对这组基的组合系数（Routing Coefficients）。
    *   **线性组合**：通过加权求和动态生成补丁特定的投影矩阵 $W_p$，实现参数的“按需分配”。
*   **混合插入策略**：为了平衡性能与稳定性，SNG 和 DPP 仅插入在各编码器阶段的最后一个块，前序块保留静态适配器，确保特征提取的稳定性。

### 4. 方法对比分析
*   **本质区别**：与传统 PEFT（如 LoRA、Adapter）不同，PointTPA 在推理时参数是“动态生成”的，而非训练后固定的。
*   **创新贡献**：引入了基于空间序列化和动态参数基组合的架构，实现了轻量化且高精度的场景级特征适配。
*   **适用场景**：适用于各类大规模、几何复杂的 3D 场景感知任务（如室内场景分割）。

### 5. 实验分析
*   **验证方法**：在 ScanNet、ScanNet++ 和 S3DIS 数据集上，对比主流 PEFT 及全量微调（FFT）。
*   **关键结论**：在仅使用 1.09% 的参数情况下，PointTPA 在 ScanNet 验证集上达到了 78.4% mIoU，优于 PointGST 等 SOTA 方法，且训练/推理速度提升约 4 倍。
*   **优势**：极高的参数效率（<2%）和极强的推理时动态适应能力。
*   **局限**：在极端复杂的场景下，空间填充曲线的序列化效果可能存在上限，且对动态参数基数的超参数选择敏感。

### 6. 实用指南
*   **开源情况**：代码已开源（[GitHub链接](https://github.com/H-EmbodVis/PointTPA)）。
*   **关键细节**：$s$ (缩放因子) 建议设为 1.0；DPP 建议应用在下投影层（down-projection）以最大化性能；空间填充曲线（SFC）的选择对性能有直接影响，建议使用 Mixed 策略。
*   **迁移建议**：本方法天然适合需要处理长序列或多尺度几何特征的任务，可通过替换主干网络（如从 PTv3 迁移到其他 Transformer 架构）直接使用。

### 7. 总结
*   **核心思想**：通过输入驱动的动态参数投影，实现推理时的场景级自适应。
*   **速记版 pipeline**：
    1.  用空间填充曲线将点云序列化。
    2.  划分局部补丁并池化。
    3.  利用路由机制从参数基中生成补丁特有权重。
    4.  通过这些动态权重实时调整主干网络投影。

**Key Findings:**

- We propose PointTPA, a Test-time Parameter Adaptation framework that generates input-aware network parameters for scene-level point clouds.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04933v1)
- [arXiv](https://arxiv.org/abs/2604.04933v1)

---

<a id='2604.04931v1'></a>
## [LoMa: Local Feature Matching Revisited](https://arxiv.org/abs/2604.04931v1)

**Authors:** David Nordström, Johan Edstedt, Georg Bökman, Jonathan Astermark, Anders Heyden, Viktor Larsson, Mårten Wadenbäck, Michael Felsberg, Fredrik Kahl

**Published:** 2026-04-06

**Categories:** cs.CV

**Abstract:**

Local feature matching has long been a fundamental component of 3D vision systems such as Structure-from-Motion (SfM), yet progress has lagged behind the rapid advances of modern data-driven approaches. The newer approaches, such as feed-forward reconstruction models, have benefited extensively from scaling dataset sizes, whereas local feature matching models are still only trained on a few mid-sized datasets. In this paper, we revisit local feature matching from a data-driven perspective. In our approach, which we call LoMa, we combine large and diverse data mixtures, modern training recipes, scaled model capacity, and scaled compute, resulting in remarkable gains in performance. Since current standard benchmarks mainly rely on collecting sparse views from successful 3D reconstructions, the evaluation of progress in feature matching has been limited to relatively easy image pairs. To address the resulting saturation of benchmarks, we collect 1000 highly challenging image pairs from internet data into a new dataset called HardMatch. Ground truth correspondences for HardMatch are obtained via manual annotation by the authors. In our extensive benchmarking suite, we find that LoMa makes outstanding progress across the board, outperforming the state-of-the-art method ALIKED+LightGlue by +18.6 mAA on HardMatch, +29.5 mAA on WxBS, +21.4 (1m, 10$^\circ$) on InLoc, +24.2 AUC on RUBIK, and +12.4 mAA on IMC 2022. We release our code and models publicly at https://github.com/davnords/LoMa.

**Analysis:**

以下是针对论文《LoMa: Local Feature Matching Revisited》的深度分析：

### 1. 摘要翻译
局部特征匹配长期以来是三维视觉系统（如运动恢复结构 SfM）的核心组件，但其进展已滞后于现代数据驱动方法。较新的模型（如前馈重建模型）通过扩充数据集规模受益匪浅，而特征匹配模型仍局限于中等规模数据集。本文从数据驱动视角重构了局部特征匹配。我们提出了 **LoMa**，通过结合大规模多样化数据混合、现代训练范式、扩展模型容量及算力，实现了性能的显著提升。针对现有基准测试因依赖成功重建而导致的饱和问题，我们整理了 1000 对来自互联网的极具挑战性的图像对，构建了 **HardMatch** 数据集。在广泛的评估中，LoMa 在 HardMatch 上较现有最优方法 ALIKED+LightGlue 提升了 18.6 mAA，并在多个视觉定位和匹配基准上刷新了记录。

### 2. 方法动机分析
*   **驱动力**：局部特征匹配常被质疑在深层神经网络及大规模数据驱动的“前馈重建模型”面前已过时。作者旨在通过“大力出奇迹”的扩展策略（Scaling Law），证明基于特征匹配的经典范式依然具有极高的竞争力和生命力。
*   **现有痛点**：
    1.  **数据瓶颈**：现有匹配模型训练数据过于单一（常仅用 MegaDepth）。
    2.  **基准饱和**：传统匹配测试集（如 MegaDepth-1500）对当前模型过于简单，无法体现真实场景中的鲁棒性差异。
*   **研究假设**：局部特征匹配的性能瓶颈并非算法结构本身，而是训练数据的多样性与模型容量，只要通过大规模多样化数据和增大参数规模，就能超越昂贵的稠密匹配与前馈模型。

### 3. 方法设计详解
*   **流程总结**：遵循经典的“检测+描述+匹配”三段式范式，但不训练检测器，而是固定使用 DaD 检测器。
    1.  **描述子学习**：基于 DeDoDe 架构，使用大规模数据混合进行 dual-softmax 损失训练。
    2.  **匹配器学习**：基于 LightGlue 架构，通过 transformer 自注意力与交叉注意力层精化特征。
    3.  **分层监督**：在 matcher 的每一层引入 dual-softmax 损失与匹配度预测损失，支持推理时的“早停”（early stopping）以平衡性能与延迟。
*   **关键改进**：
    *   **Data Scaling**：将训练集从单一数据集扩展至 17 个涵盖室内、室外、航拍、图形渲染等多源数据集。
    *   **Capacity Scaling**：设计了 Base、Large、Gigantic 三种不同参数规模的匹配器（Embedding 维度分别为 256, 512, 1024）。

### 4. 方法对比分析
*   **本质区别**：LoMa 并非发明一种全新的匹配架构，而是将现代大规模预训练和模型缩放技术（Scaling Laws）成功迁移至局部特征匹配领域。
*   **创新贡献**：构建了 HardMatch 数据集，这不仅是一个评估集，更是未来研究“硬匹配”问题的指南针；证明了分层监督匹配器在推理时的灵活性。

### 5. 实验分析
*   **关键结果**：在 HardMatch 上实现大幅领先，证明在极端视角变化、季节光照差异下，LoMa 比前馈重建方法更具鲁棒性。
*   **局限**：对极端的“分身”（Doppelgängers，指外观极度相似的建筑）场景仍面临挑战；在大规模描述子训练中观察到一定的过拟合倾向。

### 6. 实用指南
*   **开源地址**：[github.com/davnords/LoMa](https://github.com/davnords/LoMa)
*   **实现要点**：
    1.  **数据策略**：训练数据混合（Data Mix）是核心，必须包含不同传感器和极端基线的数据。
    2.  **超参数**：推理时推荐使用 $L=9$ 层以获取最高精度，实时应用可采用 $L=3$ 获得性能与速度的极致平衡。
*   **迁移建议**：该架构可直接作为任何 SfM 流程中特征提取模块的替换方案。

### 7. 总结
*   **核心思想**：通过大规模数据缩放与分层设计重振经典特征匹配范式。
*   **速记版pipeline**：
    1.  **多源数据混合**：引入 17 个数据集进行预训练。
    2.  **分层匹配器**：采用多层 Transformer 并行监督。
    3.  **灵活推理**：通过早停机制动态权衡精度与计算开销。

**Key Findings:**

- The newer approaches, such as feed-forward reconstruction models, have benefited extensively from scaling dataset sizes, whereas local feature matching models are still only trained on a few mid-sized datasets.
- In our approach, which we call LoMa, we combine large and diverse data mixtures, modern training recipes, scaled model capacity, and scaled compute, resulting in remarkable gains in performance.
- To address the resulting saturation of benchmarks, we collect 1000 highly challenging image pairs from internet data into a new dataset called HardMatch.
- In our extensive benchmarking suite, we find that LoMa makes outstanding progress across the board, outperforming the state-of-the-art method ALIKED+LightGlue by +18.6 mAA on HardMatch, +29.5 mAA on WxBS, +21.4 (1m, 10$^\circ$) on InLoc, +24.2 AUC on RUBIK, and +12.4 mAA on IMC 2022.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04931v1)
- [arXiv](https://arxiv.org/abs/2604.04931v1)

---

<a id='2604.04925v1'></a>
## [Fully Procedural Synthetic Data from Simple Rules for Multi-View Stereo](https://arxiv.org/abs/2604.04925v1)

**Authors:** Zeyu Ma, Alexander Raistrick, Jia Deng

**Published:** 2026-04-06

**Categories:** cs.CV

**Abstract:**

In this paper, we explore the design space of procedural rules for multi-view stereo (MVS). We demonstrate that we can generate effective training data using SimpleProc: a new, fully procedural generator driven by a very small set of rules using Non-Uniform Rational Basis Splines (NURBS), as well as basic displacement and texture patterns. At a modest scale of 8,000 images, our approach achieves superior results compared to manually curated images (at the same scale) sourced from games and real-world objects. When scaled to 352,000 images, our method yields performance comparable to--and in several benchmarks, exceeding--models trained on over 692,000 manually curated images. The source code and the data are available at https://github.com/princeton-vl/SimpleProc.

**Analysis:**

作为计算机视觉领域的专家，我对这篇论文的分析如下：

### 1. 论文核心贡献总结
该论文提出了一种名为 **SimpleProc** 的完全过程化（Fully Procedural）合成数据生成框架，用于多视图立体视觉（MVS）任务。研究证明，仅通过极少数简单的 NURBS 样条曲线及位移与纹理规则，即可生成高质量的训练数据，且在数据规模较小的情况下表现优于手动策划的数据，在大规模下可比肩甚至超越现有的海量手动数据集。

### 2. 关键创新与方法论
*   **“极简主义”的数据生成范式**：不同于以往依赖复杂游戏引擎或昂贵扫描资产（如 ScanNet 等）的方法，SimpleProc 仅通过极其精简的几何规则和过程化纹理进行建模。
*   **利用 NURBS 的几何表达**：通过 NURBS（非均匀有理 B 样条）这一数学工具，能够以数学描述的方式高效生成多样化、平滑且具有真实几何特征的物体。
*   **数据效率的跃升**：研究核心在于验证了“合成数据的多样性与质量比规模更重要”。通过极少的生成规则（而非人工标注或扫描），它能够以一半的图像规模达到甚至超越传统手动数据集的精度。

### 3. 对领域的潜在影响
*   **打破“数据饥渴”瓶颈**：MVS 领域长期受限于高质量地面真值（Ground Truth）的获取难度。SimpleProc 提供了一种低成本、可无限扩展的数据生成方案，极大地降低了训练高性能 MVS 模型的门槛。
*   **从“数据搬运”到“规则建模”的范式转移**：该工作可能会促使研究重心从收集和清理真实世界的数据集，转向研究如何设计更有效的规则来生成符合真实物理规律的合成数据（Data-centric AI 的一个重要分支）。

### 4. 相关领域与应用价值
*   **三维重建与SLAM**：直接受益于更高精度的 MVS 模型，提升环境感知能力。
*   **自动驾驶与机器人导航**：通过过程化生成的特殊场景（如极端天气、纹理缺失表面），可以补充真实数据中难以获取的边缘案例（Edge Cases）。
*   **AR/VR 内容生成**：该方法中使用的程序化建模思路可直接应用于虚拟场景资产的快速生成。
*   **通用几何学习**：基于 NURBS 的生成方式为学习底层的 3D 几何先验提供了良好的训练环境。

### 5. 可推断的潜在局限性
*   **域差异（Domain Gap）**：虽然合成数据在某些任务上表现出色，但完全过程化的物体可能缺乏真实世界中复杂的语义背景、光照干扰或非理想材质（如半透明、镜面反射），在处理极端复杂的真实场景时可能仍存在鲁棒性问题。
*   **几何多样性限制**：尽管 NURBS 具有强大的表达能力，但如果“简单的规则”无法覆盖某些特定类别的几何特征（例如具有极其细碎结构的植物或复杂机械结构），该生成器可能会遇到泛化上限。
*   **对下游任务的适配性**：目前主要针对 MVS，如果迁移到语义分割或实例检测等更依赖复杂上下文语义的任务，仅靠简单规则生成的几何结构可能不足以提供足够的信息支撑。

**专家总结：** 这篇论文的趣味性在于它极大地挑战了“合成数据越接近现实越好”的直觉。它通过数学上的“简约美”，证明了在 MVS 任务中，受控的过程化几何结构足以替代繁琐的人工数据策划，这为未来构建大规模自动化数据集开辟了一条极具潜力的道路。

**Key Findings:**

- We demonstrate that we can generate effective training data using SimpleProc: a new, fully procedural generator driven by a very small set of rules using Non-Uniform Rational Basis Splines (NURBS), as well as basic displacement and texture patterns.
- At a modest scale of 8,000 images, our approach achieves superior results compared to manually curated images (at the same scale) sourced from games and real-world objects.
- When scaled to 352,000 images, our method yields performance comparable to--and in several benchmarks, exceeding--models trained on over 692,000 manually curated images.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04925v1)
- [arXiv](https://arxiv.org/abs/2604.04925v1)

---

<a id='2604.04924v1'></a>
## [Your Pre-trained Diffusion Model Secretly Knows Restoration](https://arxiv.org/abs/2604.04924v1)

**Authors:** Sudarshan Rajagopalan, Vishal M. Patel

**Published:** 2026-04-06

**Categories:** cs.CV, cs.AI

**Abstract:**

Pre-trained diffusion models have enabled significant advancements in All-in-One Restoration (AiOR), offering improved perceptual quality and generalization. However, diffusion-based restoration methods primarily rely on fine-tuning or Control-Net style modules to leverage the pre-trained diffusion model's priors for AiOR. In this work, we show that these pre-trained diffusion models inherently possess restoration behavior, which can be unlocked by directly learning prompt embeddings at the output of the text encoder. Interestingly, this behavior is largely inaccessible through text prompts and text-token embedding optimization. Furthermore, we observe that naive prompt learning is unstable because the forward noising process using degraded images is misaligned with the reverse sampling trajectory. To resolve this, we train prompts within a diffusion bridge formulation that aligns training and inference dynamics, enforcing a coherent denoising path from noisy degraded states to clean images. Building on these insights, we introduce our lightweight learned prompts on the pre-trained WAN video model and FLUX image models, converting them into high-performing restoration models. Extensive experiments demonstrate that our approach achieves competitive performance and generalization across diverse degradations, while avoiding fine-tuning and restoration-specific control modules.

**Analysis:**

作为计算机视觉领域的专家，我对这篇论文《Your Pre-trained Diffusion Model Secretly Knows Restoration》的分析如下：

### 1. 核心贡献摘要
该论文证明了预训练扩散模型（如WAN和FLUX）内嵌了强大的图像恢复先验，且无需进行繁重的全模型微调或引入额外的控制模块（如ControlNet）。作者提出了一种通过学习文本编码器输出端的提示嵌入（Prompt Embeddings）的方法，成功“解锁”了模型底层的恢复能力，实现了高效、通用的多任务图像/视频恢复。

### 2. 关键创新与方法论
*   **Prompt Tuning 为核心机制**：研究发现传统的文本提示词难以激活恢复能力，通过直接在文本编码器的输出空间学习提示嵌入（Learned Prompts），可以更精确地引导模型的去噪行为。
*   **弥合分布差异（Bridging the Gap）**：针对“前向加噪过程与反向采样轨迹不匹配”的问题，作者引入了**扩散桥（Diffusion Bridge）**公式。通过该框架对齐训练与推理的动态，强制建立从“受损图像”到“清晰图像”的一致性去噪路径，解决了直接使用扩散模型进行恢复时常见的训练不稳定问题。
*   **无权重更新的恢复（Weight-Free Restoration）**：该方法仅优化提示空间，保留了预训练扩散模型的所有权重，这在计算资源开销和模型灵活性上具有极大的优势。

### 3. 对领域的潜在影响
*   **范式转变**：该研究挑战了“高质量恢复必须依赖复杂微调或外挂模块”的传统观点。它暗示了预训练的大规模生成模型本身就是强大的“通用感知与恢复引擎”。
*   **资源效率**：通过极低参数量的提示学习，即可赋予基础模型专业的任务能力，这将大大降低工业界部署特定恢复任务（如去噪、超分、去模糊）的门槛。
*   **模块化组合**：由于无需修改权重，该方法可以轻松适配不同的基座模型（如视频模型WAN和图像模型FLUX），极大地增强了模型的复用性。

### 4. 相关领域与应用前景
*   **低资源环境下的实时图像处理**：适用于移动端或边缘计算设备，因为不需要进行大规模参数推理。
*   **多任务通用修复（AiOR）**：在处理多种混合退化（如同时存在噪声、低分辨率、模糊）的场景中，这种方法能更好地利用预训练模型的泛化先验。
*   **视频修复与增强**：文中特别提到了WAN视频模型，这意味着该技术在老旧视频修复、低质量视频质量提升领域有广阔的应用前景。
*   **可控内容创作**：提示学习的方式可能进一步结合语义引导，实现“针对性增强”，例如在恢复过程中强调特定物体的细节。

### 5. 可推断的局限性
*   **对基座模型的依赖性**：由于该方法是“激活”而非“重构”，如果预训练模型本身的先验中未涵盖某种极端的退化类型，提示学习的效果可能会受限。
*   **扩散桥的收敛性**：虽然引入了扩散桥来对齐动态，但在处理极度复杂的非线性退化时，提示空间的搜索空间是否足够大以找到最优解仍有待观察。
*   **推理延迟**：尽管避免了权重更新，但作为扩散模型，其本质仍然是多步采样（Sampling-based），在实时性要求极高的场景（如视频实时流）中，相比传统的非扩散类（GAN或CNN）方法可能仍存在计算瓶颈。

### 专家点评
这篇论文非常有趣，因为它巧妙地揭示了**大型生成模型（Foundation Models）不仅是“创作者”，更是“修补者”**。它将焦点从“如何训练模型完成任务”转移到了“如何更好地沟通与激活模型潜力”上。对于计算机视觉从业者而言，这种无需修改核心权重即可赋予模型新任务能力的思路，是实现高效大模型应用落地的典范路径。

**Key Findings:**

- In this work, we show that these pre-trained diffusion models inherently possess restoration behavior, which can be unlocked by directly learning prompt embeddings at the output of the text encoder.
- Building on these insights, we introduce our lightweight learned prompts on the pre-trained WAN video model and FLUX image models, converting them into high-performing restoration models.
- Extensive experiments demonstrate that our approach achieves competitive performance and generalization across diverse degradations, while avoiding fine-tuning and restoration-specific control modules.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04924v1)
- [arXiv](https://arxiv.org/abs/2604.04924v1)

---

<a id='2604.04913v1'></a>
## [A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](https://arxiv.org/abs/2604.04913v1)

**Authors:** Tommie Kerssies, Gabriele Berton, Ju He, Qihang Yu, Wufei Ma, Daan de Geus, Gijs Dubbelman, Liang-Chieh Chen

**Published:** 2026-04-06

**Categories:** cs.CV

**Abstract:**

Anticipating diverse future states is a central challenge in video world modeling. Discriminative world models produce a deterministic prediction that implicitly averages over possible futures, while existing generative world models remain computationally expensive. Recent work demonstrates that predicting the future in the feature space of a vision foundation model (VFM), rather than a latent space optimized for pixel reconstruction, requires significantly fewer world model parameters. However, most such approaches remain discriminative. In this work, we introduce DeltaTok, a tokenizer that encodes the VFM feature difference between consecutive frames into a single continuous "delta" token, and DeltaWorld, a generative world model operating on these tokens to efficiently generate diverse plausible futures. Delta tokens reduce video from a three-dimensional spatio-temporal representation to a one-dimensional temporal sequence, for example yielding a 1,024x token reduction with 512x512 frames. This compact representation enables tractable multi-hypothesis training, where many futures are generated in parallel and only the best is supervised. At inference, this leads to diverse predictions in a single forward pass. Experiments on dense forecasting tasks demonstrate that DeltaWorld forecasts futures that more closely align with real-world outcomes, while having over 35x fewer parameters and using 2,000x fewer FLOPs than existing generative world models. Code and weights: https://deltatok.github.io.

**Analysis:**

### 1. 摘要翻译
预见多样化的未来状态是视频世界建模的核心挑战。判别式世界模型产生单一确定性预测，隐含地平均了所有可能的未来，而现有的生成式世界模型在计算上非常昂贵。近期研究表明，在视觉基础模型（VFM）的特征空间（而非像素重建的潜空间）中预测未来，所需的模型参数量更少，但多数此类方法仍属判别式。为此，我们提出了 **DeltaTok**，一种将连续帧间的 VFM 特征差异编码为单个连续“Delta”token 的分词器；以及 **DeltaWorld**，一个基于这些 token 高效生成多样化未来序列的生成式世界模型。Delta token 将视频从三维时空表示简化为一维时间序列，例如在 512×512 分辨率下实现了 1024 倍的 token 压缩。这种紧凑表示使得多假设训练变得易于处理——即并行生成多个未来并仅监督最佳预测，推理时可在单次前向传播中生成多样化结果。实验表明，DeltaWorld 在密集预测任务中生成的未来更符合真实结果，且参数量减少 35 倍，FLOPs 降低 2000 倍。

### 2. 方法动机分析
*   **驱动力**：旨在打破现有生成式模型“计算昂贵”与判别式模型“缺乏多样性”的矛盾，实现一种既能产生多样化未来，又极度高效（单次前向传播）的世界模型。
*   **痛点**：现有方法不仅受限于像素级重建的计算冗余，更因无法高效处理空间冗余，导致推理时需要漫长的多步采样或扩散式迭代，难以满足实时性需求。
*   **核心直觉**：视频中的连续帧具有极高的时空冗余，大部分场景是静止的。通过将预测对象从“全空间特征图”转变为“帧间的时域增量（Delta）”，可以将 3D 时空问题降维为 1D 时域序列建模。

### 3. 方法设计详解
*   **DeltaTok（差分分词器）**：
    *   这是一个基于连续自动编码器设计的模块。它接收连续两帧的 VFM 特征 ($x_{t-1}, x_t$)，输出一个单一的“Delta token” ($z_t$)。
    *   **核心逻辑**：该 token 编码了从 $x_{t-1}$ 到 $x_t$ 的转换逻辑。解码器 $h(x_{t-1}, z_t)$ 负责利用前一帧特征与 Delta token 恢复当前帧特征。
*   **DeltaWorld（生成式世界模型）**：
    *   **Best-of-Many (BoM) 训练**：模型接收不同的噪声查询（Noise Queries），在单次前向传播中并行生成 $K$ 个未来假设。仅选择 Loss 最小的那个假设进行监督，强制模型学习映射关系。
    *   **Pipeline**：
        1. 输入一系列 Delta tokens（时域序列）。
        2. 预测下一个时间步的 Delta token $\hat{z}_{t+1}$。
        3. 利用 DeltaTok 解码器将 $\hat{z}_{t+1}$ 转换为空间特征图 $\hat{x}_{t+1}$。
        4. 递归地将生成的 Delta token 加入序列以进行自回归滚动预测。

### 4. 方法对比分析
*   **本质区别**：从“空间冗余建模”转向“纯时域差异建模”。不再预测完整的下一帧像素或特征，而是预测“变化本身”。
*   **创新贡献**：提出 Delta 编码理念，实现了 1024 倍级别的 token 压缩；利用 BoM 机制，实现了无扩散（非迭代）的单步生成。
*   **适用场景**：适用于自动驾驶、视频预测等对实时性要求极高，且场景具备较强时域连续性的任务。

### 5. 实验分析
*   **结论**：DeltaWorld 在 Cityscapes 等基准测试中，以 35 倍更小的参数量和 2000 倍更低的 FLOPs，超越了大型生成式模型（如 Cosmos）。
*   **优势**：在保持判别式基线精度的同时，具备了生成多种合理未来的能力，且大幅降低了推理成本。
*   **局限**：由于采用自动回归预测 Delta token，长期预测可能存在特征漂移（累积误差）。

### 6. 实用指南
*   **开源**：代码与权重已发布至 `deltatok.github.io`。
*   **实现要点**：
    *   **训练策略**：BoM 训练中 $K$ 值的选择对多样性至关重要（文中推荐 $K=64$ 左右性能趋于稳定）。
    *   **架构选择**：基于 ViT-B 配置，使用 AdamW 优化器，线性 warmup 5K 步，后续保持较低学习率。
    *   **迁移建议**：该模块可以作为“插件”嵌入到任何基于特征空间的预测架构中，直接替换原本沉重的空间预测模块。

### 7. 总结
*   **核心思想**：用单点 Delta token 描述时空演变，以极简代价实现多未来预测。
*   **速记版pipeline**：
    1. 计算连续帧的差异（Delta）。
    2. 将差异压缩为单一 Token。
    3. 用 Transformer 并行预测多种 Delta 变体。
    4. 选出最佳变体并解码还原。

**Key Findings:**

- In this work, we introduce DeltaTok, a tokenizer that encodes the VFM feature difference between consecutive frames into a single continuous "delta" token, and DeltaWorld, a generative world model operating on these tokens to efficiently generate diverse plausible futures.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04913v1)
- [arXiv](https://arxiv.org/abs/2604.04913v1)

---

<a id='2604.04887v1'></a>
## [HorizonWeaver: Generalizable Multi-Level Semantic Editing for Driving Scenes](https://arxiv.org/abs/2604.04887v1)

**Authors:** Mauricio Soroco, Francesco Pittaluga, Zaid Tasneem, Abhishek Aich, Bingbing Zhuang, Wuyang Chen, Manmohan Chandraker, Ziyu Jiang

**Published:** 2026-04-06

**Categories:** cs.CV

**Abstract:**

Ensuring safety in autonomous driving requires scalable generation of realistic, controllable driving scenes beyond what real-world testing provides. Yet existing instruction guided image editors, trained on object-centric or artistic data, struggle with dense, safety-critical driving layouts. We propose HorizonWeaver, which tackles three fundamental challenges in driving scene editing: (1) multi-level granularity, requiring coherent object- and scene-level edits in dense environments; (2) rich high-level semantics, preserving diverse objects while following detailed instructions; and (3) ubiquitous domain shifts, handling changes in climate, layout, and traffic across unseen environments. The core of HorizonWeaver is a set of complementary contributions across data, model, and training: (1) Data: Large-scale dataset generation, where we build a paired real/synthetic dataset from Boreas, nuScenes, and Argoverse2 to improve generalization; (2) Model: Language-Guided Masks for fine-grained editing, where semantics-enriched masks and prompts enable precise, language-guided edits; and (3) Training: Content preservation and instruction alignment, where joint losses enforce scene consistency and instruction fidelity. Together, HorizonWeaver provides a scalable framework for photorealistic, instruction-driven editing of complex driving scenes, collecting 255K images across 13 editing categories and outperforming prior methods in L1, CLIP, and DINO metrics, achieving +46.4% user preference and improving BEV segmentation IoU by +33%. Project page: https://msoroco.github.io/horizonweaver/

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对 **HorizonWeaver** 这篇论文的分析如下：

### 1. 论文核心贡献总结
HorizonWeaver 提出了一种针对自动驾驶场景的通用化、多层级语义编辑框架，旨在解决现有生成式模型在处理复杂交通环境时缺乏控制力与一致性的问题。通过结合大规模跨数据集训练、语言引导的掩码机制以及内容保持的联合损失函数，该框架实现了在保持高保真度的同时，对驾驶场景进行精准且可控的语义编辑。

### 2. 关键创新与方法论
该论文的创新点在于构建了一个针对“自动驾驶垂直领域”的编辑流水线，核心包括：
*   **多层级编辑机制 (Multi-level Granularity)：** 将编辑任务分解为对象级与场景级，通过“语言引导掩码 (Language-Guided Masks)”将自然语言指令与高精度语义分割图对齐，从而实现对特定交通参与者或背景环境的精准定位与操作。
*   **跨数据源泛化 (Domain Generalization)：** 利用 Boreas、nuScenes 和 Argoverse2 等主流自动驾驶数据集构建的大规模配对数据集，打破了以往模型在特定数据集上的局限性。
*   **双重优化目标 (Content & Instruction Alignment)：** 引入联合训练损失，在遵循指令（Instruction Fidelity）的同时，通过内容保持机制（Content Preservation）确保编辑后的场景在物理和空间逻辑上与原图保持一致，避免了常见的伪影或场景错位问题。

### 3. 对领域的潜在影响
*   **数据合成与增强的范式转变：** 自动驾驶领域一直受限于“长尾效应”（即极端安全工况数据稀缺）。HorizonWeaver 能够通过编辑现有真实场景生成无限的、符合逻辑的安全工况，从而大幅降低现实中测试的高成本和风险。
*   **增强对下游任务的支持：** 论文提到的 BEV（鸟瞰图）分割 IoU 提升 +33%，证明了该技术不仅是“视觉美化”，还能有效改善感知系统的鲁棒性，这为生成式 AI 赋能感知决策系统提供了有力证明。

### 4. 相关领域与受益应用
*   **自动驾驶仿真平台 (Simulation Testing)：** 为云端大规模仿真测试提供自动化的场景生成工具，支持对特定天气、突发路况或交通密度进行“修改”。
*   **智能交通基础设施：** 视频监控分析系统可以利用此类技术对复杂路口进行模拟测试，优化交通流调度算法。
*   **多模态生成研究：** 探索如何在高动态、空间约束极强的场景中引入语言控制，这对机器人视觉控制（Robotic Vision Control）及数字孪生领域具有借鉴意义。

### 5. 可推断的局限性
*   **物理合理性约束 (Physical Plausibility)：** 虽然摘要强调了“内容保持”，但在大规模图像编辑中，模型如何保证编辑后的场景严格符合物理交通规则（例如：修改后的车辆轨迹是否符合动力学限制）是一个潜在难题。
*   **实时性瓶颈：** 考虑到该模型涉及语言解码与复杂的图像生成过程，对于需要实时生成或交互的在线系统而言，其推理延迟可能是一个需要解决的挑战。
*   **长期一致性 (Temporal Consistency)：** 摘要主要聚焦于静态图像，若将该方法推广到驾驶视频编辑中，如何保证帧间的一致性（防止闪烁）将是进一步研究的重点。

**总结：** HorizonWeaver 的有趣之处在于它成功将“通用生成式 AI”的能力降维应用到“高度结构化的自动驾驶领域”。它证明了通过特定领域的先验知识（如 BEV 空间、语义地图）注入，可以让生成模型在极高严谨性的场景下发挥作用。

**Key Findings:**

- We propose HorizonWeaver, which tackles three fundamental challenges in driving scene editing: (1) multi-level granularity, requiring coherent object- and scene-level edits in dense environments; (2) rich high-level semantics, preserving diverse objects while following detailed instructions; and (3) ubiquitous domain shifts, handling changes in climate, layout, and traffic across unseen environments.
- The core of HorizonWeaver is a set of complementary contributions across data, model, and training: (1) Data: Large-scale dataset generation, where we build a paired real/synthetic dataset from Boreas, nuScenes, and Argoverse2 to improve generalization; (2) Model: Language-Guided Masks for fine-grained editing, where semantics-enriched masks and prompts enable precise, language-guided edits; and (3) Training: Content preservation and instruction alignment, where joint losses enforce scene consistency and instruction fidelity.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04887v1)
- [arXiv](https://arxiv.org/abs/2604.04887v1)

---

<a id='2604.04874v1'></a>
## [Free-Range Gaussians: Non-Grid-Aligned Generative 3D Gaussian Reconstruction](https://arxiv.org/abs/2604.04874v1)

**Authors:** Ahan Shabanov, Peter Hedman, Ethan Weber, Zhengqin Li, Denis Rozumny, Gael Le Lan, Naina Dhingra, Lei Luo, Andrea Vedaldi, Christian Richardt, Andrea Tagliasacchi, Bo Zhu, Numair Khan

**Published:** 2026-04-06

**Categories:** cs.CV

**Abstract:**

We present Free-Range Gaussians, a multi-view reconstruction method that predicts non-pixel, non-voxel-aligned 3D Gaussians from as few as four images. This is done through flow matching over Gaussian parameters. Our generative formulation of reconstruction allows the model to be supervised with non-grid-aligned 3D data, and enables it to synthesize plausible content in unobserved regions. Thus, it improves on prior methods that produce highly redundant grid-aligned Gaussians, and suffer from holes or blurry conditional means in unobserved regions. To handle the number of Gaussians needed for high-quality results, we introduce a hierarchical patching scheme to group spatially related Gaussians into joint transformer tokens, halving the sequence length while preserving structure. We further propose a timestep-weighted rendering loss during training, and photometric gradient guidance and classifier-free guidance at inference to improve fidelity. Experiments on Objaverse and Google Scanned Objects show consistent improvements over pixel and voxel-aligned methods while using significantly fewer Gaussians, with large gains when input views leave parts of the object unobserved.

**Analysis:**

这是一份关于《Free-Range Gaussians: Non-Grid-Aligned Generative 3D Gaussian Reconstruction》的深度技术分析报告。

---

### 1. 摘要翻译
我们提出了 Free-Range Gaussians，一种多视图重建方法，它能从少至四张图像中预测非像素、非体素对齐的 3D 高斯分布。该方法通过高斯参数上的流匹配（Flow Matching）实现。我们的生成式重建框架允许使用非网格对齐的 3D 数据进行监督，并能在未观测区域合成合理的 3D 内容。这改进了现有方法——现有方法往往产生高度冗余的网格对齐高斯分布，并在未观测区域出现空洞或模糊的条件均值。为了处理高保真结果所需的高斯数量，我们引入了一种分层补丁方案，将空间相关的高斯分布组合为联合 Transformer Token，在保持结构的同时将序列长度减半。我们进一步提出了训练时的时序加权渲染损失，以及推理时的光度梯度引导和无分类器引导（Classifier-Free Guidance）以提升保真度。在 Objaverse 和 Google Scanned Objects 数据集上的实验表明，该方法在显著减少高斯数量的同时，始终优于像素和体素对齐方法，尤其是在输入视图导致物体部分未被观测时表现出巨大优势。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决现有 feed-forward 3D 重建方法在处理 sparse-view 输入时，对未观测区域重建效果不佳（要么产生空洞，要么模糊）且表示冗余的痛点。
*   **现有方法痛点**：
    *   **像素/体素对齐限制**：强行将高斯分布锚定在输入图像像素或固定 3D 体素网格上，导致表示冗余且难以适应生成式任务。
    *   **确定性回归的局限**：传统回归模型预测的是条件期望 $E[G|I]$，在未观测区域容易得到“模糊的均值”。
*   **研究假设**：通过引入生成式建模（流匹配）来学习 3D 高斯分布的条件分布 $P(G|I)$，并解除对像素/体素网格的依赖，能够实现既忠实于观测数据又能在未观测区域进行高质量“补全”的重建。

---

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **输入编码**：使用 DINOv2 提取输入视图特征，拼接位置编码、下采样 RGB 和 Plücker 射线坐标。
    2.  **流匹配（Flow Matching）**：在潜在空间对高斯参数（均值、旋转、比例、不透明度、颜色）进行迭代去噪，从标准正态分布转化为目标 3D 高斯分布。
    3.  **层次化补丁（Hierarchical Patchification）**：构建基于 LoD 的二叉树，将相邻（亲缘）的高斯分布合并为单一 Transformer Token，实现序列压缩。
    4.  **引导式推理**：在推理阶段引入光度梯度引导（将渲染误差反向传播到高斯参数）和无分类器引导（增强条件一致性）。
*   **算法核心**：利用 $z_t = (1-t)\epsilon + t z_1$ 定义线性概率路径，模型直接预测干净样本 $\hat{z}_1$。利用 timestep-weighted loss $w(t)$，在后期去噪阶段加大光度渲染监督，确保细节。

---

### 4. 方法对比与创新
*   **本质区别**：从“确定性回归”转向“生成式建模”，并彻底去除了对输入空间（像素/体素）的强结构化约束。
*   **创新贡献**：
    1.  **层次化 patchification**：解决了 Transformer 处理海量高斯分布时的计算瓶颈，同时维持空间局部性（Spatial Locality）。
    2.  **重构引导流匹配**：结合了生成模型的补全能力与基于重构的引导机制，在 FID（分布质量）和 PSNR（重构精度）上取得了双赢。

---

### 5. 实验分析
*   **验证**：在 Objaverse 和 GSO 上对比了 LGM、LaRa、GS-LRM 等前沿方法。
*   **关键结论**：在仅用 8K 高斯（相比基线的 45K-500K）的情况下，在完全观测和部分观测场景均表现出极佳的感知质量（低 FID，高 DINO similarity）。
*   **优势/局限**：优势是补全质量高、紧凑；局限是 50 步的迭代去噪比单步前馈模型慢（约 26 秒/物体）。

---

### 6. 实用指南
*   **实现细节**：
    *   **数据构建**：需预先通过 3DGS-MCMC 优化得到地面真值高斯集，并构建 KD-tree 以支持 LoD。
    *   **引导缩放**：超参数 $\lambda_{PG}=50$ 是平衡重构精度的关键。
*   **迁移建议**：该流匹配框架可直接迁移到任何支持隐式表示的对象重建任务，如几何补全或跨模态生成。

---

### 7. 总结
*   **核心思想**：利用非网格对齐的流匹配，实现兼顾输入忠实度与幻觉补全的生成式 3D 重建。
*   **速记版 Pipeline**：
    1. 提取多视图特征并将其投射为 Token；
    2. 通过层次化二叉树对高斯参数进行补丁化压缩；
    3. 利用 Transformer 在流匹配空间执行迭代去噪；
    4. 结合光度梯度引导与加权渲染损失优化输出。

**Key Findings:**

- We present Free-Range Gaussians, a multi-view reconstruction method that predicts non-pixel, non-voxel-aligned 3D Gaussians from as few as four images.
- To handle the number of Gaussians needed for high-quality results, we introduce a hierarchical patching scheme to group spatially related Gaussians into joint transformer tokens, halving the sequence length while preserving structure.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04874v1)
- [arXiv](https://arxiv.org/abs/2604.04874v1)

---

<a id='2604.04857v1'></a>
## [The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](https://arxiv.org/abs/2604.04857v1)

**Authors:** Runhao Mao, Hanshi Wang, Yixiang Yang, Qianli Ma, Jingmeng Zhou, Zhipeng Zhang

**Published:** 2026-04-06

**Categories:** cs.CV

**Abstract:**

The integration of Vision-Language Models (VLMs) into autonomous driving promises to solve long-tail scenarios, but this paradigm faces the critical and unaddressed challenge of catastrophic forgetting. The very fine-tuning process used to adapt these models to driving-specific data simultaneously erodes their invaluable pre-trained world knowledge, creating a self-defeating paradox that undermines the core reason for their use. This paper provides the first systematic investigation into this phenomenon. We introduce a new large-scale dataset of 180K scenes, which enables the first-ever benchmark specifically designed to quantify catastrophic forgetting in autonomous driving. Our analysis reveals that existing methods suffer from significant knowledge degradation. To address this, we propose the Drive Expert Adapter (DEA), a novel framework that circumvents this trade-off by shifting adaptation from the weight space to the prompt space. DEA dynamically routes inference through different knowledge experts based on scene-specific cues, enhancing driving-task performance without corrupting the model's foundational parameters. Extensive experiments demonstrate that our approach not only achieves state-of-the-art results on driving tasks but also effectively mitigates catastrophic forgetting, preserving the essential generalization capabilities that make VLMs a transformative force for autonomous systems. Data and model are released at FidelityDrivingBench.

**Analysis:**

这是一份关于《The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models》的深度技术分析。

### 1. 摘要翻译
视觉语言模型（VLM）在自动驾驶中的应用有望解决长尾场景，但这一范式面临着严重的灾难性遗忘挑战。微调过程在赋予模型驾驶专业知识的同时，侵蚀了其宝贵的预训练世界知识，造成了与其初衷背道而驰的矛盾。本文首次对这一现象进行了系统研究。我们提出了一个包含18万个场景的大规模数据集，建立了首个量化自动驾驶中灾难性遗忘的基准。分析表明，现有微调方法存在显著的知识衰退。为此，我们提出了“驾驶专家适配器”（Drive Expert Adapter, DEA），该框架通过将适配从权重空间转移到提示空间（Prompt Space）来规避这一权衡。DEA根据场景提示动态路由推理至不同的知识专家，在不破坏模型基础参数的前提下提升了驾驶任务性能。

### 2. 方法动机分析
*   **驱动力**：利用预训练VLM的强大泛化能力解决自动驾驶长尾场景，但必须解决微调带来的“知识遗忘”悖论。
*   **痛点**：全量微调（Full Fine-tuning）导致模型严重偏离预训练的通用认知；轻量化方案（如LoRA）虽缓解了遗忘，但往往无法很好地桥接通用领域与驾驶领域之间的巨大Gap，导致在极端长尾场景下性能不升反降。
*   **研究假设**：通过“提示空间”而非“权重空间”进行任务适配，并结合动态专家路由，可以实现知识保存与任务性能提升的平衡。

### 3. 方法设计详解
**Pipeline流程**：
1.  **输入与感知**：模型接收单目图像，结合LLM处理任务指令。
2.  **提示适配（Prompt Adapter）**：模型不直接修改基础参数，而是利用学习到的可训练Embedding（Prompt Tokens）预置到输入中，编码场景先验。
3.  **动态专家路由（TAEM）**：构建一个动态MoE（Mixture-of-Experts）系统。通过一个轻量级门控网络（Gating Network），根据场景 cues（如天气、交通密度）和提示信息，动态激活最匹配的LoRA专家模块。
4.  **推理生成**：在固定基础模型（Frozen VLM）的基础上，融合上述适配器的输出产生最终决策。

*   **模型结构**：分为冻结的基础VLM、提示适配器（PA）、任务自适应专家模块（TAEM）三部分。PA提供全局场景适配，TAEM提供局部的专家级场景处理。
*   **算法本质**：将“参数微调”转化为“推理期的条件计算（Conditional Computing）”，通过控制信息流向而非权重，实现对预训练知识的“防破坏保护”。

### 4. 方法对比分析
*   **本质区别**：从传统的“修改参数以适应任务”转变为“固定核心参数，动态路由任务需求”。
*   **创新点**：提出了基于IDF频率引导的自动挖掘Pipeline，构建了首个针对自动驾驶领域量化遗忘的基准（Fidelity Driving Bench）。
*   **适用场景**：适用于资源受限、既需要保留通用感知能力又需要特定驾驶专业度的高安全性自动驾驶决策系统。

### 5. 实验分析
*   **验证方法**：在18万帧场景上通过Knowledge Retention Rate (KRR) 和 Noteworthy Objects’ Perception Recall (NoPR) 指标评估。
*   **关键结果**：DEA框架在保持KRR（79.0%）大幅领先的同时，实现了对传统全量微调方法在场景描述和交通问答任务上的性能超越。
*   **优劣势**：优势在于极其强大的鲁棒性和对长尾物体的识别率；局限在于引入了额外的专家模块，推理开销较纯单体模型略有增加。

### 6. 实用指南
*   **开源情况**：已发布于 `FidelityDrivingBench`。
*   **实现要点**：
    *   **提示工程**：需确保提示适配器学到的Embedding对特定任务家族的区分度。
    *   **门控网络**：训练时需关注门控网络的负载均衡，避免特定专家过拟合，需引入负载损失函数。
*   **迁移建议**：DEA的设计范式可直接移植到其他需保护预训练知识的垂直领域，只需更换对应的“专家库”即可。

### 7. 总结
*   **核心思想**：通过提示适配与动态专家路由，在不重写权重的前提下实现专业化适配。
*   **速记版pipeline**：
    1.  收集多源驾驶数据；
    2.  IDF过滤提取长尾高价值场景；
    3.  部署提示适配器编码通用先验；
    4.  通过门控机制动态路由专业LoRA专家；
    5.  输出最终的驾驶行为描述与决策。

**Key Findings:**

- We introduce a new large-scale dataset of 180K scenes, which enables the first-ever benchmark specifically designed to quantify catastrophic forgetting in autonomous driving.
- To address this, we propose the Drive Expert Adapter (DEA), a novel framework that circumvents this trade-off by shifting adaptation from the weight space to the prompt space.
- Extensive experiments demonstrate that our approach not only achieves state-of-the-art results on driving tasks but also effectively mitigates catastrophic forgetting, preserving the essential generalization capabilities that make VLMs a transformative force for autonomous systems.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04857v1)
- [arXiv](https://arxiv.org/abs/2604.04857v1)

---

<a id='2604.04834v1'></a>
## [E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes](https://arxiv.org/abs/2604.04834v1)

**Authors:** Jiajun Zhai, Hao Shi, Shangwei Guo, Kailun Yang, Kaiwei Wang

**Published:** 2026-04-06

**Categories:** cs.CV, cs.MM, cs.RO, eess.IV

**Abstract:**

Robotic Vision-Language-Action (VLA) models generalize well for open-ended manipulation, but their perception is fragile under sensing-stage degradations such as extreme low light, motion blur, and black clipping. We present E-VLA, an event-augmented VLA framework that improves manipulation robustness when conventional frame-based vision becomes unreliable. Instead of reconstructing images from events, E-VLA directly leverages motion and structural cues in event streams to preserve semantic perception and perception-action consistency under adverse conditions. We build an open-source teleoperation platform with a DAVIS346 event camera and collect a real-world synchronized RGB-event-action manipulation dataset across diverse tasks and illumination settings. We also propose lightweight, pretrained-compatible event integration strategies and study event windowing and fusion for stable deployment. Experiments show that even a simple parameter-free fusion, i.e., overlaying accumulated event maps onto RGB images, could substantially improve robustness in dark and blur-heavy scenes: on Pick-Place at 20 lux, success increases from 0% (image-only) to 60% with overlay fusion and to 90% with our event adapter; under severe motion blur (1000 ms exposure), Pick-Place improves from 0% to 20-25%, and Sorting from 5% to 32.5%. Overall, E-VLA provides systematic evidence that event-driven perception can be effectively integrated into VLA models, pointing toward robust embodied intelligence beyond conventional frame-based imaging. Code and dataset will be available at https://github.com/JJayzee/E-VLA.

**Analysis:**

### 1. 摘要翻译
机器人视觉-语言-动作（VLA）模型在开放式操作任务中表现出色，但在极端低光、运动模糊和信号截断等感知退化条件下显得脆弱。我们提出了 E-VLA，这是一个事件驱动的 VLA 框架，旨在提高传统基于图像的系统在不利条件下的操作鲁棒性。E-VLA 不通过事件重建图像，而是直接利用事件流中的运动和结构特征来保持语义感知与动作的一致性。我们构建了基于 DAVIS346 相机的开源远程操作平台，采集了同步的 RGB-事件-动作数据集。此外，我们提出了轻量级、预训练兼容的事件集成策略。实验表明，简单的事件叠加融合可显著提升暗光场景下的鲁棒性，而我们提出的层级式事件适配器（Event Adapter）在处理严重运动模糊和低光任务时表现卓越，为鲁棒具身智能提供了一种超越传统成像的新路径。

### 2. 方法动机分析
*   **驱动力**：解决 VLA 模型在“感官退化”（极端光照、快速运动）环境下的感知失效问题。
*   **痛点**：传统 RGB 摄像头在低光下信号信噪比极低，高速运动下产生严重运动模糊，且图像增强或重建算法无法恢复丢失的原始信息，且通常伴随高计算延迟。
*   **研究假设**：通过异步、高动态范围的事件流（Event Stream）补充 RGB 的空间语义，利用事件的高时间分辨率捕捉运动轨迹，能够弥补 frame-based 视觉在极端环境下的信息缺失。

### 3. 方法设计详解
*   **处理流程**：
    1.  **事件窗口化（Recent-count Windowing）**：针对机器人操作动作快慢不一导致的非平稳事件流，抛弃传统的固定时间窗口，采用“最近 $N$ 个事件”的动态计数窗口，确保感知响应的一致性。
    2.  **图像化表示**：将窗口内的事件投影并去马赛克，转化为与 RGB 图像分布一致的 3 通道灰度图 $E$。
    3.  **融合策略**：
        *   **Overlay 融合（参数无关）**：在编码前将事件图直接叠加到 RGB 图像上。
        *   **层级式适配器（Learnable）**：在 SigLIP 编码器的中间层注入事件特征。通过共享权重的 Patch Embedding 层和 Transformer 块，将事件信息逐级融合到图像编码器的特征映射中。
    4.  **模型架构**：冻结 VLM（SmolVLM）主干，仅微调事件适配器和动作专家层。

### 4. 方法对比与创新
*   **本质区别**：不将事件流视为独立的补丁输入，而是作为视觉编码器内部的“辅助分支”，或是前置的像素级融合，避免了庞大的 token 序列带来的计算负担。
*   **创新贡献**：提出了一种与现有的、Frozen 的大规模 VLA 架构完美适配的“即插即用”方案，且对原有预训练知识的破坏性降到最低。

### 5. 实验分析
*   **核心结论**：在 20 lux 极端暗光下，Image-only 方法成功率降至 0%，而 E-VLA 叠加策略达 60%，适配器变体高达 90%。
*   **鲁棒性**：在 severe motion blur (1000ms 曝光) 下，E-VLA 依然能保持有效动作输出，显著优于单纯的图像增强方法。
*   **局限性**：对需要精确色彩语义的任务（如 Sorting）在完全失真环境下存在瓶颈，因为事件流本质上对颜色信息不敏感。

### 6. 实用指南
*   **开源情况**：代码和数据集已公开，基于 LeRobot 格式。
*   **训练细节**：
    *   两阶段训练：先训练适配器（冻结 VLM），再联合微调。
    *   Dropout：在训练时对图像分支应用 50% 的随机 Dropout，以强制模型学习依赖事件流特征，防止“捷径学习”（Shortcut Learning）。
*   **迁移**：该适配器设计非常轻量（13M 参数），可轻易迁移到任何基于 ViT 编码器的 VLA 模型（如 RT-2, OpenVLA）。

### 7. 总结
*   **核心思想**：通过轻量级事件适配器，将异步运动感知无缝嵌入主流 VLA 架构。
*   **速记版 Pipeline**：
    1. 实时采集事件流并执行动态事件计数窗口化；
    2. 将事件转化为像素一致的图像表示；
    3. 通过层级适配器将事件特征“注入”预训练视觉编码器；
    4. 在保留 VLM 核心知识的前提下，微调策略输出。

**Key Findings:**

- We present E-VLA, an event-augmented VLA framework that improves manipulation robustness when conventional frame-based vision becomes unreliable.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04834v1)
- [arXiv](https://arxiv.org/abs/2604.04834v1)

---

<a id='2604.04811v1'></a>
## [AnyUser: Translating Sketched User Intent into Domestic Robots](https://arxiv.org/abs/2604.04811v1)

**Authors:** Songyuan Yang, Huibin Tan, Kailun Yang, Wenjing Yang, Shaowu Yang

**Published:** 2026-04-06

**Categories:** cs.RO, cs.CV, cs.HC

**Abstract:**

We introduce AnyUser, a unified robotic instruction system for intuitive domestic task instruction via free-form sketches on camera images, optionally with language. AnyUser interprets multimodal inputs (sketch, vision, language) as spatial-semantic primitives to generate executable robot actions requiring no prior maps or models. Novel components include multimodal fusion for understanding and a hierarchical policy for robust action generation. Efficacy is shown via extensive evaluations: (1) Quantitative benchmarks on the large-scale dataset showing high accuracy in interpreting diverse sketch-based commands across various simulated domestic scenes. (2) Real-world validation on two distinct robotic platforms, a statically mounted 7-DoF assistive arm (KUKA LBR iiwa) and a dual-arm mobile manipulator (Realman RMC-AIDAL), performing representative tasks like targeted wiping and area cleaning, confirming the system's ability to ground instructions and execute them reliably in physical environments. (3) A comprehensive user study involving diverse demographics (elderly, simulated non-verbal, low technical literacy) demonstrating significant improvements in usability and task specification efficiency, achieving high task completion rates (85.7%-96.4%) and user satisfaction. AnyUser bridges the gap between advanced robotic capabilities and the need for accessible non-expert interaction, laying the foundation for practical assistive robots adaptable to real-world human environments.

**Analysis:**

以下是对论文《AnyUser: Translating Sketched User Intent into Domestic Robots》的深入技术分析：

### 1. 摘要翻译
AnyUser是一个统一的机器人指令系统，通过在环境照片上进行自由手绘草图（可选配语言指令），实现直观的家庭任务指定。AnyUser将多模态输入（草图、视觉、语言）解释为空间语义基元，从而生成无需预先构建地图或模型的机器人可执行动作。系统创新性地采用了多模态融合理解和鲁棒动作生成的层级策略。广泛评估证明：1）在模拟环境基准中，该系统对各种草图指令的解释准确率极高；2）在真实物理环境（KUKA 7-DoF机械臂与Realman双臂移动操作平台）中验证了其指令落地能力；3）涵盖不同人群（老年人、非语言沟通者等）的用户研究表明，AnyUser显著提升了交互易用性与任务效率（完成率85.7%-96.4%）。

### 2. 方法动机分析
- **驱动力**：旨在填补高级机器人能力与普通用户（特别是老人或非技术用户）之间巨大的操作鸿沟。
- **现有方法痛点**：自然语言指令处理复杂空间关系模糊；视觉编程依赖预置地图，难以应对动态家庭场景；端到端学习方法严重依赖海量特定环境数据，且通用性差。
- **研究假设**：手绘草图作为一种“空间脚手架”，能将用户的意图直接锚定在视觉空间中，通过多模态融合，可以构建出无需复杂环境建模的泛化机器人控制策略。

### 3. 方法设计详解
- **核心Pipeline**：
  1.  **输入与预处理**：获取场景照片$I$，用户草图$S$，语言描述$L$。草图通过曲率阈值（$\theta_{turn}$）和长度约束（$L_{max}$）被分解为有序的基元序列$S_{seq}$。
  2.  **多模态编码**：使用视觉Transformer ($\phi_V$)提取场景语义，草图编码器 ($\phi_S$)识别关键点序列，语言编码器 ($\phi_L$)处理意图约束。
  3.  **多模态融合 ($\psi_{fuse}$)**：利用跨模态注意力机制将草图特征关联至视觉补丁，生成空间接地的任务表达 $R$。
  4.  **分层策略 ($\pi_{HL}$)**：基于 $R$ 输出一系列离散的宏动作（如 forward, turn, cover_area 等）。
  5.  **动作执行与转换 ($g_{translate}$)**：将抽象宏动作映射为具体的机器人控制信号（关节速度或末端位姿），并结合实时感知 $P_t$ 进行闭环修正（如避障、Under-Obstacle操作）。

### 4. 方法对比分析
- **本质区别**：AnyUser不依赖3D地图，而是将“草图-图像”作为核心参考系，将2D交互的语义直接映射为3D动作。
- **创新贡献**：提出了一种将自由手绘轨迹转化为空间语义基元的多模态理解框架；构建了混合数据策略（实拍+合成），有效提升了系统对不同家庭环境的泛化能力。

### 5. 实验分析（精简版）
- **验证方法**：利用HouseholdSketch大数据集进行仿真评测，并在KUKA和Realman平台上进行真实部署及跨人群用户研究。
- **关键结果**：单步动作成功率 (SSSR) 约 84.4%，FTCR 随任务复杂度增加呈衰减趋势，但对复杂任务（如清扫家具下空间）表现稳健。
- **主要优势**：极低的交互门槛，无需建图的即时响应能力，对复杂路径和区域覆盖的理解力强。
- **主要局限**：对超长视距的任务，随着执行步数增加，姿态累积误差（Pose drift）成为主要瓶颈。

### 6. 实用指南
- **开源与数据**：HouseholdSketch包含35,000+条标注任务，可作为后续视觉-空间机器人交互研究的基准。
- **实现细节**：
  - **超参数**：$\theta_{turn}=30^\circ$, $L_{max}=0.5m$, $d_{step}=0.05m$。
  - **避障关键**：通过 $P_t$ 输入门控机制实现，仅在障碍物检测时动态注入 egocentric 特征。
- **迁移建议**：对于其他机器人底盘，仅需重写 $g_{translate}$ 模块将宏动作解译为特定动力学控制指令即可。

### 7. 总结
- **核心思想**：手绘草图提供空间语义锚点，融合图像与实时感知进行闭环动作决策。
- **速记版pipeline**：草图分割 -> 多模态融合 -> 高层动作序列预测 -> 闭环执行与避障。

**Key Findings:**

- We introduce AnyUser, a unified robotic instruction system for intuitive domestic task instruction via free-form sketches on camera images, optionally with language.
- Novel components include multimodal fusion for understanding and a hierarchical policy for robust action generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.04811v1)
- [arXiv](https://arxiv.org/abs/2604.04811v1)

---

