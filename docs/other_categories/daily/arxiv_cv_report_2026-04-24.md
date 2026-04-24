time: 20260424

# Arxiv Computer Vision Papers - 2026-04-24

## Executive Summary

以下是为您准备的每日报告执行摘要，涵盖2026年4月23日发表在Arxiv上的10篇计算机视觉论文。摘要旨在帮助忙碌的研究人员快速把握领域内的重要发展。

---

### 每日计算机视觉论文执行摘要 (2026-04-23)

**1. 主要主题与趋势概览**

本日论文呈现出三大核心趋势：

- **3D视觉与机器人操作的深度融合**：多篇论文聚焦于如何利用3D表示（如点云、4D点云）和视图合成技术，提升机器人在复杂、非结构化环境中的感知与操作能力。这体现了从“看”到“做”的强烈工程导向。
- **多模态大模型的边界探索与问题暴露**：研究不仅关注如何构建更强大的通用视觉模型（如基于图像生成器），也深入剖析了现有大型视觉语言模型（LVLMs）的严重缺陷，特别是提示词引发的幻觉问题。这标志着领域正从“造工具”转向“理解工具”。
- **视频与场景编辑的精细化与自动化**：视频重拍（Reshooting）成为一个热点，出现了基于4D点云、自监督等不同范式的解决方案。同时，通过人类-AI协同构建精确视频语言的工作，体现了对高质量、精细控制数据标注的追求。

**2. 特别重要或创新的论文**

- **《Sapiens2》 (Rawal Khirodkar et al.)**：作为“Sapiens”系列的延续，这篇论文很可能在人体建模、姿态估计或渲染方面取得了重大突破。鉴于前作的影响力，Sapiens2很可能成为该领域的新基准，值得所有从事人体相关研究的读者关注。
- **《When Prompts Override Vision: Prompt-Induced Hallu-cinations in LVLMs》 (Pegah Khayatan et al.)**：这项工作具有高度的警示意义。它系统性地揭示了提示词如何“覆盖”视觉输入，导致模型产生看似合理但完全错误的回答。这是对当前LVLMs安全性和可靠性的一次严肃拷问，对模型部署和评估具有直接指导价值。
- **《Image Generators are Generalist Vision Learners》 (Valentin Gabeur et al.)**：这篇论文挑战了传统范式，提出将图像生成模型本身用作通用的视觉任务学习者。这是一种极具潜力的“反向”思路，如果成立，可能革新我们对视觉预训练和目标任务的认知。

**3. 新兴研究方向与技术点**

- **“无提议”（Proposal-Free）的3D实例分割**：SpaCeFormer摒弃了传统的“先检测再分割”两步法，直接进行开放词汇的3D实例分割。这在实时性和处理未知类别方面可能具备优势，是3D感知领域的一个重要尝试。
- **“轨迹条件”（Trace-Conditioned）的机器人规划**：Long-Horizon Manipulation论文引入轨迹作为条件，引导VLA（视觉-语言-动作）模型进行长期操作规划。这超越了简单的“下一步”预测，旨在实现更连贯、更复杂的任务执行。
- **上下文展开（Context Unrolling）**：在Omni Models中，通过逐步展开或回溯上下文来处理长序列或复杂关系，这可能是一种提升多模态模型处理长程依赖能力的通用技巧。
- **自监督视频重拍**：Reshoot-Anything提出完全无需标注数据的自监督方案，这对于处理海量、多样化的“野生”视频素材是极为实用的技术路线。

**4. 建议精读的论文（按推荐优先级）**

1.  **《When Prompts Override Vision...》**: **强推**。对于任何使用或评估LVLMs的研究者，理解其幻觉机制至关重要。这篇论文提供了清晰的实验证据和分类，是了解模型局限性的必读材料。
2.  **《Image Generators are Generalist Vision Learners》**: **强推**。一篇可能引发讨论和范式转变的论文。即使不同意其观点，其方法和实验设计也极具启发性，适合对通用视觉模型感兴趣的研究者。
3.  **《Sapiens2》**: **强推**。如果你是人体视觉（Human Vision）方向的研究者，应优先阅读，以期快速跟进可能的新SOTA。
4.  **《SpaCeFormer》** 与 **《Long-Horizon Manipulation》**: **按需选读**。前者适合3D视觉和开放词汇研究者；后者对机器人操作和具身智能领域的读者价值极高。
5.  **《Vista4D》** 与 **《Reshoot-Anything》**: **按需选读**。对视频编辑、图形学和3D重建感兴趣的读者可以从这两篇论文中了解最新的技术方案。

---
**总结**：本日论文展现了计算机视觉领域的成熟与自我审视。一方面，3D、机器人、视频编辑等应用方向在技术上不断深化；另一方面，社区正投入更多精力去理解和纠正底层模型（尤其是LVLMs）的固有缺陷。这种“建设”与“反思”并行的状态，是技术健康发展的标志。

---

## Table of Contents

1. [Sapiens2](#2604.21681v1)
2. [SpaCeFormer: Fast Proposal-Free Open-Vocabulary 3D Instance Segmentation](#2604.20395v1)
3. [Building a Precise Video Language with Human-AI Oversight](#2604.21718v1)
4. [Image Generators are Generalist Vision Learners](#2604.20329v1)
5. [Long-Horizon Manipulation via Trace-Conditioned VLA Planning](#2604.21924v1)
6. [Context Unrolling in Omni Models](#2604.21921v1)
7. [Vista4D: Video Reshooting with 4D Point Clouds](#2604.21915v1)
8. [VistaBot: View-Robust Robot Manipulation via Spatiotemporal-Aware View Synthesis](#2604.21914v1)
9. [When Prompts Override Vision: Prompt-Induced Hallucinations in LVLMs](#2604.21911v1)
10. [Reshoot-Anything: A Self-Supervised Model for In-the-Wild Video Reshooting](#2604.21776v1)

---

## Papers

<a id='2604.21681v1'></a>
## [Sapiens2](https://arxiv.org/abs/2604.21681v1)

**Authors:** Rawal Khirodkar, He Wen, Julieta Martinez, Yuan Dong, Su Zhaoen, Shunsuke Saito

**Published:** 2026-04-23

**Categories:** cs.CV

**Abstract:**

We present Sapiens2, a model family of high-resolution transformers for human-centric vision focused on generalization, versatility, and high-fidelity outputs. Our model sizes range from 0.4 to 5 billion parameters, with native 1K resolution and hierarchical variants that support 4K. Sapiens2 substantially improves over its predecessor in both pretraining and post-training. First, to learn features that capture low-level details (for dense prediction) and high-level semantics (for zero-shot or few-label settings), we combine masked image reconstruction with self-distilled contrastive objectives. Our evaluations show that this unified pretraining objective is better suited for a wider range of downstream tasks. Second, along the data axis, we pretrain on a curated dataset of 1 billion high-quality human images and improve the quality and quantity of task annotations. Third, architecturally, we incorporate advances from frontier models that enable longer training schedules with improved stability. Our 4K models adopt windowed attention to reason over longer spatial context and are pretrained with 2K output resolution. Sapiens2 sets a new state-of-the-art and improves over the first generation on pose (+4 mAP), body-part segmentation (+24.3 mIoU), normal estimation (45.6% lower angular error) and extends to new tasks such as pointmap and albedo estimation. Code: https://github.com/facebookresearch/sapiens2

**Analysis:**

作为计算机视觉领域的专家，我为您对 **Sapiens2** 的分析如下：

### 1. 主要贡献总结
Sapiens2 是一个专为人体视觉任务设计的高分辨率 Transformer 模型族（0.4B 至 5B 参数），通过统一的预训练目标和大规模高质量数据，显著提升了人体姿态、语义分割及表面属性估计的精度。该模型支持从原生 1K 到 4K 的多分辨率输出，在多项人体分析任务中刷新了 SOTA，并扩展了人体点云与反照率估计等新能力。

### 2. 关键创新与方法论
*   **统一预训练目标：** 该模型结合了**掩码图像重建 (Masked Image Reconstruction)** 与**自蒸馏对比学习 (Self-distilled Contrastive Objectives)**。这种策略同时兼顾了密集预测任务所需的低级特征细节和零样本/少样本任务所需的语义表征。
*   **架构优化与扩展性：** 采用了前沿模型的设计理念，增强了长训练周期的稳定性。针对 4K 高分辨率任务，引入了**窗口注意力机制 (Windowed Attention)**，能够有效处理大规模空间上下文，并采用 2K 输出分辨率进行预训练以实现平滑过渡。
*   **数据质量与规模：** 利用 10 亿规模的高质量人体图像集进行预训练，并精细优化了下游任务的标注质量，实现了从“规模化”到“精细化”的协同提升。

### 3. 对计算机视觉领域的潜在影响
*   **通用人体基础模型 (Generalist Foundation Models)：** Sapiens2 证明了单一的基础模型架构可以通过统一预训练覆盖广泛的人体感知任务，这标志着人体理解从“特定任务小模型”向“通用基础大模型”范式的深入转化。
*   **高保真度感知标准：** 该模型在法向估计（Normal Estimation）和反照率估计（Albedo Estimation）上的突破，暗示了计算机视觉正从简单的“检测”转向“基于物理的全身三维重建与材质感知”。
*   **计算效率与效能：** 通过在不同规模（0.4B-5B）上实现高性能，为工业界在资源受限与高性能需求场景下提供了灵活性。

### 4. 相关应用领域
*   **虚拟现实与元宇宙：** 高精度人体分割与姿态估计是高质量数字人驱动、动作捕捉及沉浸式体验的基础。
*   **AR 增强现实与时尚科技：** 利用点云与反照率估计，可以实现更真实的虚拟试穿（Virtual Try-on）和光照一致的 AR 遮挡（Occlusion）效果。
*   **自动驾驶与机器人：** 对行人进行高精度语义分析及三维空间建模，是提升复杂环境下机器人交互与路径规划能力的核心。
*   **医疗健康：** 在运动康复评估或远程医疗中，高保真的姿态分析和人体形态估计具有极高的临床价值。

### 5. 可推断的局限性
*   **计算资源门槛：** 尽管模型有较小版本，但达到 SOTA 效果的 5B 参数版本及 4K 推理需求对部署设备的 GPU 算力和显存提出了较高要求。
*   **数据依赖性：** 性能高度依赖于那 10 亿张高质量图像，模型在非受控环境（如低光照、极端遮挡或非典型人体姿态）下的泛化能力是否会受到训练数据偏见（Dataset Bias）的影响仍需评估。
*   **实时性能瓶颈：** 虽然采用了窗口注意力机制，但在极高分辨率（4K）下的推理延迟可能仍难以完全满足实时交互场景（如 60 FPS 的 AR 实时渲染），可能需要在精度与速度之间做出权衡。

**专家视角评价：**
Sapiens2 的趣味性在于它不仅是在“刷榜”，而是通过**预训练任务设计的范式迁移**（结合对比学习与重建），成功解决了人体视觉任务中常见的“语义理解与细节感知不可兼得”的问题。它代表了计算机视觉领域将人体理解作为“统一几何与语义表征”的趋势。

**Key Findings:**

- We present Sapiens2, a model family of high-resolution transformers for human-centric vision focused on generalization, versatility, and high-fidelity outputs.
- Sapiens2 sets a new state-of-the-art and improves over the first generation on pose (+4 mAP), body-part segmentation (+24.3 mIoU), normal estimation (45.6% lower angular error) and extends to new tasks such as pointmap and albedo estimation.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.21681v1)
- [arXiv](https://arxiv.org/abs/2604.21681v1)

---

<a id='2604.20395v1'></a>
## [SpaCeFormer: Fast Proposal-Free Open-Vocabulary 3D Instance Segmentation](https://arxiv.org/abs/2604.20395v1)

**Authors:** Chris Choy, Junha Lee, Chunghyun Park, Minsu Cho, Jan Kautz

**Published:** 2026-04-22

**Categories:** cs.CV, cs.RO

**Abstract:**

Open-vocabulary 3D instance segmentation is a core capability for robotics and AR/VR, but prior methods trade one bottleneck for another: multi-stage 2D+3D pipelines aggregate foundation-model outputs at hundreds of seconds per scene, while pseudo-labeled end-to-end approaches rely on fragmented masks and external region proposals. We present SpaCeFormer, a proposal-free space-curve transformer that runs at 0.14 seconds per scene, 2-3 orders of magnitude faster than multi-stage 2D+3D pipelines. We pair it with SpaCeFormer-3M, the largest open-vocabulary 3D instance segmentation dataset (3.0M multi-view-consistent captions over 604K instances from 7.4K scenes) built through multi-view mask clustering and multi-view VLM captioning; it reaches 21x higher mask recall than prior single-view pipelines (54.3% vs 2.5% at IoU > 0.5). SpaCeFormer combines spatial window attention with Morton-curve serialization for spatially coherent features, and uses a RoPE-enhanced decoder to predict instance masks directly from learned queries without external proposals. On ScanNet200 we achieve 11.1 zero-shot mAP, a 2.8x improvement over the prior best proposal-free method; on ScanNet++ and Replica, we reach 22.9 and 24.1 mAP, surpassing all prior methods including those using multi-view 2D inputs.

**Analysis:**

作为计算机视觉领域的专家，我对 **SpaCeFormer** 这篇论文的分析如下：

### 1. 核心贡献总结
SpaCeFormer 提出了一种高效、无提案（proposal-free）的开放词汇 3D 实例分割框架，在保持高性能的同时，实现了 0.14 秒/场景的极速推理。该研究通过引入大规模数据集 SpaCeFormer-3M，从根本上解决了现有 3D 分割方法在效率与泛化能力之间的权衡难题，在多个基准测试中大幅超越了现有的最优方案。

### 2. 关键创新与方法论
该论文的核心技术突破在于**将空间结构建模与序列化建模相结合**：
*   **空间曲线变换器（Space-Curve Transformer）：** 创新地利用 Morton 曲线（空间填充曲线）将 3D 点云序列化，结合空间窗口注意力机制（Spatial Window Attention），在保持 3D 几何空间一致性的同时，显著降低了计算复杂度。
*   **RoPE 增强解码器：** 引入旋转位置编码（RoPE）来处理查询（Queries），能够直接从特征空间预测实例掩码，彻底摆脱了传统方法对繁琐的外部区域建议（Region Proposals）的依赖。
*   **大规模监督信号：** 构建了 SpaCeFormer-3M 数据集，通过多视图掩码聚类和 VLM（视觉语言模型）标注，大幅提升了对物体边界的感知能力和召回率。

### 3. 对领域的潜在影响
*   **重新定义 3D 分割的实时性标准：** 将推理速度提升了 2-3 个数量级，使得实时交互式 3D 环境理解成为可能，不再受限于离线处理。
*   **去提案化趋势：** 证明了“无需提案”的端到端框架同样可以在开放词汇任务中取得 SOTA 效果，这将促使研究人员放弃复杂的 multi-stage 2D+3D 流水线，转而寻求更简洁的架构。
*   **数据驱动的范式转变：** SpaCeFormer-3M 的发布为 3D 视觉领域提供了一个类似 ImageNet 或 COCO 的大规模高质量基准，有助于缩小 3D 领域与 2D 大模型在零样本迁移能力上的差距。

### 4. 关联领域与应用前景
*   **机器人感知：** 极高的推理速度使其非常适合家用服务机器人、自动驾驶等需要实时决策的移动平台，实现对环境中任意物体的即时识别与抓取。
*   **AR/VR：** 在空间计算设备中，能够快速分割场景并对虚拟物体进行遮挡与交互处理。
*   **数字孪生：** 大规模场景下的快速语义地图构建，提升了环境理解的效率。
*   **多模态大模型（LMMs）：** 为具备“空间意识”的 3D 多模态大模型提供了底层的语义分割模块，使模型能更好地理解 3D 几何特征。

### 5. 可推断的潜在局限性
*   **序列化的信息丢失：** 虽然 Morton 曲线有效，但在处理高度复杂、非凸或非密集分布的 3D 结构时，线性化操作可能导致局部几何关系的信息损失，难以完全等同于真正的 3D 点云卷积。
*   **对 VLM 质量的依赖：** 尽管数据集规模庞大，但其标注质量本质上依赖于预训练 VLM 的输出，若 VLM 对特定罕见类别或几何遮挡严重区域的理解存在偏差，该偏差会直接遗传给 SpaCeFormer。
*   **内存开销与长序列：** 随着场景规模（如整栋楼宇）的增大，Transformer 处理长序列时的二次方复杂度限制依然是瓶颈，文中并未提及该模型在超大场景下的可扩展性。

**专家点评：**
SpaCeFormer 的吸引力在于它成功“啃下了” 3D 视觉中最难的两块硬骨头：**速度**与**开放词汇的泛化**。在当前多模态模型逐渐向 3D 领域延伸的背景下，这种能与大模型对齐的轻量化感知架构，极有可能是未来机器人感知系统的标配雏形。

**Key Findings:**

- We present SpaCeFormer, a proposal-free space-curve transformer that runs at 0.14 seconds per scene, 2-3 orders of magnitude faster than multi-stage 2D+3D pipelines.
- On ScanNet200 we achieve 11.1 zero-shot mAP, a 2.8x improvement over the prior best proposal-free method; on ScanNet++ and Replica, we reach 22.9 and 24.1 mAP, surpassing all prior methods including those using multi-view 2D inputs.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.20395v1)
- [arXiv](https://arxiv.org/abs/2604.20395v1)

---

<a id='2604.21718v1'></a>
## [Building a Precise Video Language with Human-AI Oversight](https://arxiv.org/abs/2604.21718v1)

**Authors:** Zhiqiu Lin, Chancharik Mitra, Siyuan Cen, Isaac Li, Yuhan Huang, Yu Tong Tiffany Ling, Hewei Wang, Irene Pi, Shihang Zhu, Ryan Rao, George Liu, Jiaxi Li, Ruojin Li, Yili Han, Yilun Du, Deva Ramanan

**Published:** 2026-04-22

**Categories:** cs.CV, cs.AI, cs.CL, cs.LG, cs.MM

**Abstract:**

Video-language models (VLMs) learn to reason about the dynamic visual world through natural language. We introduce a suite of open datasets, benchmarks, and recipes for scalable oversight that enable precise video captioning. First, we define a structured specification for describing subjects, scenes, motion, spatial, and camera dynamics, grounded by hundreds of carefully defined visual primitives developed with professional video creators such as filmmakers. Next, to curate high-quality captions, we introduce CHAI (Critique-based Human-AI Oversight), a framework where trained experts critique and revise model-generated pre-captions into improved post-captions. This division of labor improves annotation accuracy and efficiency by offloading text generation to models, allowing humans to better focus on verification. Additionally, these critiques and preferences between pre- and post-captions provide rich supervision for improving open-source models (Qwen3-VL) on caption generation, reward modeling, and critique generation through SFT, DPO, and inference-time scaling. Our ablations show that critique quality in precision, recall, and constructiveness, ensured by our oversight framework, directly governs downstream performance. With modest expert supervision, the resulting model outperforms closed-source models such as Gemini-3.1-Pro. Finally, we apply our approach to re-caption large-scale professional videos (e.g., films, commercials, games) and fine-tune video generation models such as Wan to better follow detailed prompts of up to 400 words, achieving finer control over cinematography including camera motion, angle, lens, focus, point of view, and framing. Our results show that precise specification and human-AI oversight are key to professional-level video understanding and generation. Data and code are available on our project page: https://linzhiqiu.github.io/papers/chai/

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇论文的分析如下：

### 1. 主要贡献总结
该论文提出了一个通过“人机协同监督”（Human-AI Oversight）构建精确视频描述语言的系统框架，旨在解决视频理解中描述模糊、缺乏结构化的问题。通过引入 CHAI（基于批判的监督机制）和一套视觉原语，作者大幅提升了视频理解模型的描述精度，并成功将该方法应用于视频生成模型的精细化控制（如运镜、视角等），性能超越了 Gemini-3.1-Pro 等顶尖闭源模型。

### 2. 关键创新与方法论
*   **结构化视觉原语（Visual Primitives）：** 不同于以往通用、笼统的描述，论文定义了涵盖主体、场景、运动、空间关系及镜头语言（Camera Dynamics）的视觉原语。这种“结构化规范”为视频理解设定了明确的参照系。
*   **CHAI（Critique-based Human-AI Oversight）框架：** 改变了传统的“人类从头编写”或“机器自动生成”的模式，采用了“模型生成初稿 -> 专家批判/修正 -> 生成高质量定稿”的迭代流水线。这种分工极大地平衡了标注效率与精准度。
*   **全流程优化方案：** 论文不仅停留在数据层面，还通过 SFT（监督微调）、DPO（直接偏好优化）以及推理时扩展（Inference-time scaling），将人工反馈转化为模型改进的闭环，实现了从模型理解到生成控制的全面增强。

### 3. 对领域的潜在影响
*   **从“描述性”到“控制性”的跨越：** 这项工作标志着视频模型从单纯的“看懂”转向了具备“专业电影制作级”的理解与控制能力。对于视频生成模型（如 Wan 模型），这种对镜头语言（Focus, Angle, Framing）的精细化指令遵循是工业界期待已久的突破。
*   **数据工程的范式转移：** 强调了“高质量监督”优于“大规模低质量数据”。证明了即便在有限的专家监督下，通过合理的反馈逻辑，也能训练出超越巨型模型性能的专用模型。
*   **闭源与开源的博弈：** 证明了开源社区（Qwen3-VL + 协同监督）完全有能力在特定专业任务上击败顶尖闭源模型，这对未来大模型应用研发具有重要的参考价值。

### 4. 相关领域与应用价值
*   **影视创作与特效行业：** 为 AI 辅助剪辑、智能分镜设计及精细化视频生成提供了技术支撑。
*   **具身智能（Embodied AI）：** 对于机器人理解复杂空间动态和视角变化至关重要，因为机器人需要在动态环境中识别“镜头”般的主体运动与空间关系。
*   **视频搜索与多媒体资产管理：** 能够实现更细粒度的视频检索（如“查找包含摇镜头且主体位于画面左下角的片段”）。

### 5. 可推断的局限性
*   **对专业知识的依赖：** 该方法依赖于电影从业者等“专家”进行监督，这意味着对于垂直领域（如医疗影像、特定工业场景）的迁移可能需要重新定义视觉原语，实施成本较高。
*   **可扩展性瓶颈：** 虽然通过分工提升了效率，但引入人工批判环节依然无法彻底摆脱对高质量人工参与的依赖，难以直接套用在 TB 级超大规模数据的自动化清洗上。
*   **计算开销：** 推理时扩展（Inference-time scaling）通常会增加推理延迟，对于实时性要求极高的应用场景可能存在挑战。

**总结：** 这篇论文的趣味性在于它将“电影语言”量化为计算图谱，并通过一套行之有效的流程证明了“人机协作”是通往精准视频理解的必由之路。对于追求模型可解释性和精细控制的研究者来说，这是一项里程碑式的实验。

**Key Findings:**

- We introduce a suite of open datasets, benchmarks, and recipes for scalable oversight that enable precise video captioning.
- Next, to curate high-quality captions, we introduce CHAI (Critique-based Human-AI Oversight), a framework where trained experts critique and revise model-generated pre-captions into improved post-captions.
- With modest expert supervision, the resulting model outperforms closed-source models such as Gemini-3.1-Pro.
- Finally, we apply our approach to re-caption large-scale professional videos (e.g., films, commercials, games) and fine-tune video generation models such as Wan to better follow detailed prompts of up to 400 words, achieving finer control over cinematography including camera motion, angle, lens, focus, point of view, and framing.
- Our results show that precise specification and human-AI oversight are key to professional-level video understanding and generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.21718v1)
- [arXiv](https://arxiv.org/abs/2604.21718v1)

---

<a id='2604.20329v1'></a>
## [Image Generators are Generalist Vision Learners](https://arxiv.org/abs/2604.20329v1)

**Authors:** Valentin Gabeur, Shangbang Long, Songyou Peng, Paul Voigtlaender, Shuyang Sun, Yanan Bao, Karen Truong, Zhicheng Wang, Wenlei Zhou, Jonathan T. Barron, Kyle Genova, Nithish Kannen, Sherry Ben, Yandong Li, Mandy Guo, Suhas Yogin, Yiming Gu, Huizhong Chen, Oliver Wang, Saining Xie, Howard Zhou, Kaiming He, Thomas Funkhouser, Jean-Baptiste Alayrac, Radu Soricut

**Published:** 2026-04-22

**Categories:** cs.CV, cs.AI

**Abstract:**

Recent works show that image and video generators exhibit zero-shot visual understanding behaviors, in a way reminiscent of how LLMs develop emergent capabilities of language understanding and reasoning from generative pretraining. While it has long been conjectured that the ability to create visual content implies an ability to understand it, there has been limited evidence that generative vision models have developed strong understanding capabilities. In this work, we demonstrate that image generation training serves a role similar to LLM pretraining, and lets models learn powerful and general visual representations that enable SOTA performance on various vision tasks. We introduce Vision Banana, a generalist model built by instruction-tuning Nano Banana Pro (NBP) on a mixture of its original training data alongside a small amount of vision task data. By parameterizing the output space of vision tasks as RGB images, we seamlessly reframe perception as image generation. Our generalist model, Vision Banana, achieves SOTA results on a variety of vision tasks involving both 2D and 3D understanding, beating or rivaling zero-shot domain-specialists, including Segment Anything Model 3 on segmentation tasks, and the Depth Anything series on metric depth estimation. We show that these results can be achieved with lightweight instruction-tuning without sacrificing the base model's image generation capabilities. The superior results suggest that image generation pretraining is a generalist vision learner. It also shows that image generation serves as a unified and universal interface for vision tasks, similar to text generation's role in language understanding and reasoning. We could be witnessing a major paradigm shift for computer vision, where generative vision pretraining takes a central role in building Foundational Vision Models for both generation and understanding.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇题为《Image Generators are Generalist Vision Learners》的论文分析如下：

### 1. 论文核心贡献总结
该研究证实了图像生成预训练模型（Generative Pretraining）能够学习到强大的通用视觉表征，其作用类似于大语言模型（LLM）的生成式预训练。作者通过将各类视觉任务（如分割、深度估计）统一重构为“RGB图像生成”问题，成功构建了一个无需牺牲生成能力的通用视觉大模型“Vision Banana”，在多项2D和3D视觉任务上达到了SOTA水平。

### 2. 关键创新与方法论
*   **统一的生成式范式（Unifying Vision Tasks as Generation）：** 这是该论文最具颠覆性的设计。它打破了以往针对特定任务（如检测、分割）设计特定头（Heads）的做法，而是将感知任务的输出空间参数化为RGB图像（例如将深度图或分割掩码看作一种特殊的视觉内容生成）。
*   **指令微调（Instruction-Tuning）：** 通过在生成模型（NBP）的基础上，使用极少量的视觉任务数据进行轻量级指令微调，使模型在保持强大生成能力的同时，具备了极强的判别性理解能力。
*   **生成即理解的范式转换：** 确立了“生成能力蕴含理解能力”的假设，证明了视觉模型可以像LLM一样，通过统一的生成预训练任务获得涌现的理解能力。

### 3. 对计算机视觉领域的潜在影响
*   **范式转移（Paradigm Shift）：** 这可能标志着计算机视觉正式进入“大模型统一架构”时代。类似于LLM统一了NLP任务，该论文提出的生成式接口可能终结多种专用视觉模型并存的碎片化格局，将生成式预训练确定为基础模型（Foundation Models）构建的核心范式。
*   **通用接口的确立：** 证明了“RGB图像”可以作为计算机视觉任务的“通用语言”，为未来多模态交互提供了一个简单、直观且强大的架构框架。

### 4. 受益的相关领域或应用
*   **自动驾驶与机器人感知：** 能够在一个模型内同时处理深度估计、语义分割、物体跟踪等任务，极大减少模型部署开销。
*   **医学影像分析：** 生成式模型对细节的捕捉能力使其在病灶分割、图像增强及病理分析方面具有巨大潜力。
*   **交互式AI设计：** 这种生成与理解一体化的模型，将大幅提升AI系统在复杂视觉场景下进行推理和操作的能力，对具身智能（Embodied AI）的进化至关重要。

### 5. 可推断的潜在限制
*   **计算成本与推理延迟：** 虽然论文提到指令微调是“轻量级”的，但将复杂的判别任务（如精细的语义分割）转化为图像生成过程，其推理开销（Inference Latency）可能远高于传统的CNN或ViT检测头，在实时性要求极高的边缘设备上可能面临挑战。
*   **准确度与可控性：** 传统的判别模型在特定任务上（如精确到像素的边界）具有严格的数学约束，而生成式模型本质上是概率性的。如何保证模型在进行视觉感知时输出结果的精确性和稳定性（即“幻觉”问题），是实际应用中必须解决的难点。
*   **训练数据的依赖性：** 尽管实现了zero-shot能力，但该模型表现的上限极度依赖于基础生成模型（NBP）在超大规模数据集上的预训练质量。

### 专家点评：
这篇论文的意义在于它**极大地简化了视觉模型的架构复杂性**。如果说“视觉Transformer (ViT)”解决的是特征提取的标准化问题，那么这篇论文解决的就是**视觉输出的标准化问题**。它证明了计算机视觉不需要专门的“任务头”，通过统一生成式接口，我们可以实现“一套权重，万般任务”，这无疑是通往视觉通用人工智能（AGI）的重要一步。

**Key Findings:**

- In this work, we demonstrate that image generation training serves a role similar to LLM pretraining, and lets models learn powerful and general visual representations that enable SOTA performance on various vision tasks.
- We introduce Vision Banana, a generalist model built by instruction-tuning Nano Banana Pro (NBP) on a mixture of its original training data alongside a small amount of vision task data.
- We show that these results can be achieved with lightweight instruction-tuning without sacrificing the base model's image generation capabilities.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.20329v1)
- [arXiv](https://arxiv.org/abs/2604.20329v1)

---

<a id='2604.21924v1'></a>
## [Long-Horizon Manipulation via Trace-Conditioned VLA Planning](https://arxiv.org/abs/2604.21924v1)

**Authors:** Isabella Liu, An-Chieh Cheng, Rui Yan, Geng Chen, Ri-Zhao Qiu, Xueyan Zou, Sha Yi, Hongxu Yin, Xiaolong Wang, Sifei Liu

**Published:** 2026-04-23

**Categories:** cs.RO

**Abstract:**

Long-horizon manipulation remains challenging for vision-language-action (VLA) policies: real tasks are multi-step, progress-dependent, and brittle to compounding execution errors. We present LoHo-Manip, a modular framework that scales short-horizon VLA execution to long-horizon instruction following via a dedicated task-management VLM. The manager is decoupled from the executor and is invoked in a receding-horizon manner: given the current observation, it predicts a progress-aware remaining plan that combines (i) a subtask sequence with an explicit done + remaining split as lightweight language memory, and (ii) a visual trace -- a compact 2D keypoint trajectory prompt specifying where to go and what to approach next. The executor VLA is adapted to condition on the rendered trace, thereby turning long-horizon decision-making into repeated local control by following the trace. Crucially, predicting the remaining plan at each step yields an implicit closed loop: failed steps persist in subsequent outputs, and traces update accordingly, enabling automatic continuation and replanning without hand-crafted recovery logic or brittle visual-history buffers. Extensive experiments spanning embodied planning, long-horizon reasoning, trajectory prediction, and end-to-end manipulation in simulation and on a real Franka robot demonstrate strong gains in long-horizon success, robustness, and out-of-distribution generalization. Project page: https://www.liuisabella.com/LoHoManip

**Analysis:**

以下是对论文《Long-Horizon Manipulation via Trace-Conditioned VLA Planning》的深度分析：

### 1. 摘要翻译
长程操纵任务对视觉-语言-动作（VLA）策略而言仍具挑战性，因为这类任务通常是多步骤、进度依赖且对执行误差极其敏感的。我们提出了LoHo-Manip，一个模块化框架，通过专用的任务管理VLM将短程VLA执行能力扩展到长程指令遵循任务。该管理器与执行器解耦，以递归方式运行：在给定当前观测的情况下，它预测一个具备进度感知能力的“剩余计划”，该计划结合了（i）作为轻量级语言记忆的“已完成+剩余”子任务序列，以及（ii）作为紧凑型2D关键点轨迹的“视觉轨迹”，用以指引下一步动作目标。执行器VLA被调整为能够根据渲染出的轨迹进行条件化决策，从而将长程决策问题转化为通过“跟随轨迹”实现的重复局部控制。

### 2. 方法动机分析
- **驱动力**：解决VLA模型在处理长程、多步骤任务时因误差累积导致的失败（即“脆性”问题）。
- **痛点**：单体化模型难以同时处理长程规划（宏观）和短程控制（微观）；长历史窗口导致训练推理失配且容易过拟合。
- **核心假设**：将长程操纵解构为“由VLM负责规划（what/where next）”和“由VLA负责执行（how to do it）”，通过引入“视觉轨迹”作为空间约束，可以实现更鲁棒的闭环控制。

### 3. 方法设计详解
LoHo-Manip的核心在于解耦后的递归管理：
- **任务管理器（Task Manager）**：
  - **输入**：当前观测 $o_t$ 和文本形式的进度记忆 $C_{t-1}$（Done: a-c; Remaining: d-f）。
  - **输出**：更新后的剩余子任务列表 $R_t^*$ 和 视觉轨迹 $\tau_t^*$（2D关键点序列）。
  - **特点**：不依赖长视觉历史，通过文本记忆维持进度感，保证稳定性。
- **执行器（Executor）**：
  - **作用**：是一个轻量级微调后的VLA（如$\pi_{0.5}$）。
  - **关键改进**：将预测的轨迹 $\tau_t^*$ 渲染在观测图像上，作为一种空间增强提示。执行器学习“紧随轨迹”这一通用技能。
- **闭环机制**：每执行完一个步骤（或固定间隔），系统重新调用管理器，根据当前状态动态更新计划，天然具备了自动纠错和重规划能力。

### 4. 方法对比分析
- **本质区别**：与端到端模型不同，它显式地将“规划（轨迹/语义）”与“控制（动作生成）”拆开。
- **创新点**：
    - **视觉轨迹（Visual Trace）**：将语义任务落地为像素坐标轨迹，作为一种直观、通用的空间引导，降低了VLA理解语义的压力。
    - **显式进度记忆（$C_t, R_t$）**：摒弃复杂的历史buffer，用简洁的文本标签实现进度追踪。
- **适用场景**：复杂、多步骤的桌面操纵任务，且对鲁棒性和泛化能力要求高的场景。

### 5. 实验分析
- **验证方法**：在RoboVQA（ reasoning）、EgoPlan-Bench（规划）、LIBERO/VLABench（操纵）上进行测试，并在真实Franka机器人上完成多步操纵任务。
- **关键结论**：在长程任务成功率和OOD（分布外）泛化性能上显著优于端到端VLA模型。
- **优势**：极强的模块化，可随意更换底层VLA执行器；显式纠错，能通过重新规划处理诸如“抓错物体”等执行错误。
- **局限**：对感知能力依赖高，如果管理器识别错误，会导致级联失败；仅限2D轨迹，难以处理极度复杂的接触动力学。

### 6. 实用指南
- **开源情况**：项目地址为 [https://www.liuisabella.com/LoHoManip](https://www.liuisabella.com/LoHoManip)。
- **实现建议**：在微调执行器时，需要构建高质量的、带有轨迹关键点标注的演示数据；管理器需具备良好的物体定位能力。
- **迁移性**：该架构可以直接套用在任何现有的VLA模型上，通过将其作为“导航员”插件引入，提升原模型的长程处理能力。

### 7. 总结
- **核心思想**：通过预测空间轨迹作为中间表示，将长程操纵解构为动态重规划与轨迹跟随。
- **速记版Pipeline**：
    1. **解析**：当前观测+任务指令传入管理器。
    2. **规划**：输出下一步子任务与对应的轨迹关键点。
    3. **渲染**：将轨迹关键点叠加到输入图像中。
    4. **执行**：VLA根据轨迹信息生成控制动作。
    5. **递归**：循环上述步骤，直到任务完成。

**Key Findings:**

- We present LoHo-Manip, a modular framework that scales short-horizon VLA execution to long-horizon instruction following via a dedicated task-management VLM.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.21924v1)
- [arXiv](https://arxiv.org/abs/2604.21924v1)

---

<a id='2604.21921v1'></a>
## [Context Unrolling in Omni Models](https://arxiv.org/abs/2604.21921v1)

**Authors:** Ceyuan Yang, Zhijie Lin, Yang Zhao, Fei Xiao, Hao He, Qi Zhao, Chaorui Deng, Kunchang Li, Zihan Ding, Yuwei Guo, Fuyun Wang, Fangqi Zhu, Xiaonan Nie, Shenhan Zhu, Shanchuan Lin, Hongsheng Li, Weilin Huang, Guang Shi, Haoqi Fan

**Published:** 2026-04-23

**Categories:** cs.CV

**Abstract:**

We present Omni, a unified multimodal model natively trained on diverse modalities, including text, images, videos, 3D geometry, and hidden representations. We find that such training enables Context Unrolling, where the model explicitly reasons across multiple modal representations before producing predictions. This process enables the model to aggregate complementary information across heterogeneous modalities, facilitating a more faithful approximation of the shared multimodal knowledge manifold and improving downstream reasoning fidelity. As a result, Omni achieves strong performance on both multimodal generation and understanding benchmarks, while demonstrating advanced multimodal reasoning capabilities, including in-context generation of text, image, video, and 3D geometry.

**Analysis:**

这是一份关于《Context Unrolling in Omni Models》的深度方法分析。

### 1. 摘要翻译
我们提出了Omni，一个在文本、图像、视频、3D几何和隐空间表示等多种模态上进行原生训练的统一多模态模型。我们发现这种训练方式能够激发“上下文展开”（Context Unrolling）能力，即模型在生成预测前，会在多个模态表征之间进行显式的推理。该过程使模型能够聚合异构模态间的互补信息，从而更忠实地逼近共享多模态知识流形，并提升下游推理的保真度。结果表明，Omni不仅在多模态生成和理解基准测试中表现强劲，还展示了先进的多模态推理能力，包括文本、图像、视频和3D几何的上下文生成。

### 2. 方法动机分析
- **驱动力**：作者试图打破现有统一模型仅将多模态任务简单堆叠在“多任务容器”中的做法，旨在实现模态间的深度交互。
- **痛点**：现有模型通常将多模态输入直接映射为输出，忽略了模态间隐含的互补信息，导致在复杂推理任务（如空间理解）中表现欠佳。
- **研究假设**：多模态知识流形在不同模态下的投影是互补的，模型通过显式地“展开”并整合这些异构投影，能构建出更完整、一致的内部世界表征。

### 3. 方法设计详解
- **核心流程**：Omni将推理视为“迭代上下文构建 + 上下文条件解码”的过程。
  - **公式**：$C_{t+1} = C_t \oplus \phi_t(x, C_t)$，其中 $\phi_t$ 是原子推理算子（如“描述”、“估计深度”、“推断相机位姿”、“渲染新视图”）。
  - **Pipeline**：模型先根据任务需求激活相关的原子模态算子，将几何约束、语义推理结果或生成的中间视觉表征写入共享的“工作空间”（Context），最后基于更新后的Context执行解码任务。
- **模型结构**：基于BAGEL的混合专家架构（MoE），包含3B活跃参数。通过扩展训练模态（增加视频、3D几何、隐表示），使其支持原生多模态上下文的构建。
- **算法意义**：该方法将“任务”重构为“上下文构建算子”，通过扩展上下文的长度、结构和语义信息，降低后续生成任务的歧义性。

### 4. 方法对比分析
- **本质区别**：传统模型关注“单向映射”，Omni关注“推理过程的可视化与结构化”，将中间推理结果显式转化为Context的一部分。
- **创新点**：提出了“Context Unrolling”这一新机制，通过定义 atomic primitives（原子原语）来操控多模态信息，将生成任务转化为受控推理过程。
- **适用场景**：复杂空间几何推理、需要多步逻辑的图像/视频生成、以及需要跨模态验证（如利用几何约束校准图像生成）的任务。

### 5. 实验分析
- **关键结果**：在MMSI-Bench空间理解测试中，加入几何/视觉上下文后的模型表现大幅优于纯文本推理；在Text-to-Image生成中，通过“思考”和视觉token辅助，Prompt遵循能力显著提升。
- **优势**：显著增强了空间智能和组合性生成能力，推理路径更透明且可校准。
- **局限**：目前视频生成的分辨率（480x640）和时长（12秒）仍落后于顶尖专用模型，计算成本随迭代深度增加。

### 6. 实用指南
- **开源情况**：项目主页为 [omni-model.com](https://omni-model.com/)，建议关注后续权重发布。
- **迁移建议**：该方法的核心在于“原子原语”的设计。若想迁移，需在模型预训练阶段引入多模态协同数据（如文本描述+对应深度图+对应的语义分割），并将推理能力融入微调阶段的思维链（CoT）训练中。
- **注意点**：需要平衡计算开销（Context长度）与推理收益，过长的上下文可能引入噪声。

### 7. 总结
- **核心思想**：将多模态推理视为构建并迭代优化共享工作空间的显式过程。
- **速记版Pipeline**：
  1. **解析需求**：根据输入任务激活特定模态算子。
  2. **上下文展开**：利用模型生成中间的语义、结构或几何线索。
  3. **条件构建**：将这些线索拼接进工作空间。
  4. **最终解码**：在完善的上下文引导下输出最终结果。

**Key Findings:**

- We present Omni, a unified multimodal model natively trained on diverse modalities, including text, images, videos, 3D geometry, and hidden representations.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.21921v1)
- [arXiv](https://arxiv.org/abs/2604.21921v1)

---

<a id='2604.21915v1'></a>
## [Vista4D: Video Reshooting with 4D Point Clouds](https://arxiv.org/abs/2604.21915v1)

**Authors:** Kuan Heng Lin, Zhizheng Liu, Pablo Salamanca, Yash Kant, Ryan Burgert, Yuancheng Xu, Koichi Namekata, Yiwei Zhao, Bolei Zhou, Micah Goldblum, Paul Debevec, Ning Yu

**Published:** 2026-04-23

**Categories:** cs.CV

**Abstract:**

We present Vista4D, a robust and flexible video reshooting framework that grounds the input video and target cameras in a 4D point cloud. Specifically, given an input video, our method re-synthesizes the scene with the same dynamics from a different camera trajectory and viewpoint. Existing video reshooting methods often struggle with depth estimation artifacts of real-world dynamic videos, while also failing to preserve content appearance and failing to maintain precise camera control for challenging new trajectories. We build a 4D-grounded point cloud representation with static pixel segmentation and 4D reconstruction to explicitly preserve seen content and provide rich camera signals, and we train with reconstructed multiview dynamic data for robustness against point cloud artifacts during real-world inference. Our results demonstrate improved 4D consistency, camera control, and visual quality compared to state-of-the-art baselines under a variety of videos and camera paths. Moreover, our method generalizes to real-world applications such as dynamic scene expansion and 4D scene recomposition. See our project page for results, code, and models: https://eyeline-labs.github.io/Vista4D

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇关于 **Vista4D** 的论文分析如下：

### 1. 主要贡献总结
Vista4D 提出了一种创新的视频重拍（Video Reshooting）框架，通过构建显式的 4D 点云（4D Point Clouds）将输入视频“锚定”在三维空间中，实现了对动态场景在任意新相机轨迹下的高保真重合成。该方法有效解决了传统方法在深度估计伪影、内容一致性保持以及复杂相机轨迹控制方面的难题，为动态视频的可编辑性和重渲染提供了强有力的几何基础。

### 2. 关键创新与方法论
*   **4D 锚定表征（4D-grounded Representation）：** 不同于仅依赖隐式神经表征的方法，Vista4D 明确构建了包含静态像素分割和 4D 重建的显式点云表征，这种“显式+隐式”结合的思路能更好地保留原始场景外观。
*   **鲁棒性训练机制：** 为了应对现实世界中 4D 重建可能存在的点云噪声和 Artifacts，研究者采用重构的多视角动态数据对模型进行训练。这显著增强了模型在面对非理想输入时的鲁棒性。
*   **精准的相机控制：** 通过在 4D 空间中建模，系统赋予了用户对输出视角和相机轨迹的精确控制权，突破了以往视频重绘方法在空间约束上的局限。

### 3. 对该领域的潜在影响
*   **从“像素生成”向“几何一致性生成”的范式转变：** 该研究表明，在视频生成和编辑领域，仅仅依靠 2D 扩散模型往往难以处理复杂的运动几何；将 4D 几何作为生成过程的“骨架”是提升视频时空一致性的关键路径。
*   **降低高质量视频制作门槛：** 这种技术使得普通视频拍摄后即可进行“后期镜头重拍”，在电影工业、广告制作以及个人内容创作中具有极高的应用价值。

### 4. 相关领域与应用场景
*   **动态场景扩展（Outpainting & Extrapolation）：** 在现有视频视角之外扩展空间范围，用于修复或扩充全景视频。
*   **4D 场景重构与虚拟漫游：** 将传统的“单视图”视频转化为可自由漫游的 4D 模型，应用于数字孪生、VR/AR 内容创作。
*   **电影工业后期特效：** 在无需重拍的情况下，导演可以通过 Vista4D 在后期修改摄像机运动（如平移、旋转、缩放），从而改变叙事节奏和视觉呈现。

### 5. 可推断的局限性
*   **点云质量依赖性：** 尽管模型对噪声具有鲁棒性，但其底层效果仍受限于原始视频的 4D 重建质量。如果输入视频中存在极其复杂的遮挡或极速运动，重建失败可能导致重合成出现崩坏。
*   **计算成本与实时性：** 构建 4D 点云表征通常涉及昂贵的计算开销（如结构从运动 SfM、深度估计等），这可能限制了其在实时交互式应用中的部署。
*   **超长视频的连续性：** 4D 点云表征在处理长时序视频时可能会面临显存爆炸或漂移问题，论文可能主要聚焦于短时片段。

---
**专家点评：**
Vista4D 的有趣之处在于它巧妙地绕过了单纯依赖“生成式人工智能”进行视频合成的不可控性，选择通过**几何先行（Geometry-first）**的手段来约束生成过程。这种“几何引导生成”的趋势（类似之前的 3D Gaussian Splatting 在视频生成中的应用）是当前计算机视觉领域解决视频一致性问题的最前沿方向之一，极具学术与工程价值。

**Key Findings:**

- We present Vista4D, a robust and flexible video reshooting framework that grounds the input video and target cameras in a 4D point cloud.
- Specifically, given an input video, our method re-synthesizes the scene with the same dynamics from a different camera trajectory and viewpoint.
- Existing video reshooting methods often struggle with depth estimation artifacts of real-world dynamic videos, while also failing to preserve content appearance and failing to maintain precise camera control for challenging new trajectories.
- Our results demonstrate improved 4D consistency, camera control, and visual quality compared to state-of-the-art baselines under a variety of videos and camera paths.
- Moreover, our method generalizes to real-world applications such as dynamic scene expansion and 4D scene recomposition.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.21915v1)
- [arXiv](https://arxiv.org/abs/2604.21915v1)

---

<a id='2604.21914v1'></a>
## [VistaBot: View-Robust Robot Manipulation via Spatiotemporal-Aware View Synthesis](https://arxiv.org/abs/2604.21914v1)

**Authors:** Songen Gu, Yuhang Zheng, Weize Li, Yupeng Zheng, Yating Feng, Xiang Li, Yilun Chen, Pengfei Li, Wenchao Ding

**Published:** 2026-04-23

**Categories:** cs.RO

**Abstract:**

Recently, end-to-end robotic manipulation models have gained significant attention for their generalizability and scalability. However, they often suffer from limited robustness to camera viewpoint changes when training with a fixed camera. In this paper, we propose VistaBot, a novel framework that integrates feed-forward geometric models with video diffusion models to achieve view-robust closed-loop manipulation without requiring camera calibration at test time. Our approach consists of three key components: 4D geometry estimation, view synthesis latent extraction, and latent action learning. VistaBot is integrated into both action-chunking (ACT) and diffusion-based ($π_0$) policies and evaluated across simulation and real-world tasks. We further introduce the View Generalization Score (VGS) as a new metric for comprehensive evaluation of cross-view generalization. Results show that VistaBot improves VGS by 2.79$\times$ and 2.63$\times$ over ACT and $π_0$, respectively, while also achieving high-quality novel view synthesis. Our contributions include a geometry-aware synthesis model, a latent action planner, a new benchmark metric, and extensive validation across diverse environments. The code and models will be made publicly available.

**Analysis:**

以下是对论文《VistaBot: View-Robust Robot Manipulation via Spatiotemporal-Aware View Synthesis》的深度解析：

### 1. 摘要翻译
近期，端到端机器人操作模型在通用性和可扩展性方面备受关注，但它们在固定相机训练后，对视角变化的鲁棒性往往较弱。本文提出了VistaBot，这是一种创新框架，通过将前馈几何模型与视频扩散模型相融合，在无需测试时相机标定的情况下，实现了视角鲁棒的闭环操作。该方法包含4D几何估计、视图合成潜变量提取和潜变量动作学习三个关键组件。VistaBot被集成到动作分块（ACT）和基于扩散的（$\pi_0$）策略中，并在仿真和真实环境中进行了评估。我们还引入了视图泛化评分（VGS）作为评估跨视角泛化的新指标。结果表明，VistaBot在VGS指标上较ACT和$\pi_0$分别提升了2.79倍和2.63倍，同时实现了高质量的视觉合成。

### 2. 方法动机分析
*   **驱动力**：打破“训练-测试视角必须一致”的铁律，实现仅需单一视角演示即可在任意视角下执行任务的泛化能力。
*   **现有痛点**：基于重构的方法（如NeRF/3DGS）计算昂贵且难以应对实时闭环要求；基于生成的方法往往缺乏对机器人动作的几何一致性理解，导致视角偏移时出现物理失真。
*   **核心假设**：通过显式的几何先验（深度与位姿）构建“结构骨架”，再利用视频扩散模型进行“语义填充”，能生成兼具物理一致性和空间精度的潜变量，从而稳固闭环策略的决策。

### 3. 方法设计详解
*   **核心Pipeline**：
    1.  **4D几何估计（VGGT）**：输入推断视角下的单帧图像，通过微调的VGGT模型预测深度图$D_n$与相对位姿$T_{n \to t}$，将点云重投影回训练视角的“基准面”。
    2.  **空间-时间潜变量提取**：利用点云重投影图像及掩码，通过Video Diffusion Model (CogVideoX) 进行修复（Inpainting）。通过在不同视角间插值相机姿态，确保多帧生成过程的平滑与时序一致性。
    3.  **闭环动作学习**：策略不再直接观测RGB图像，而是将DiT块提取的高阶潜变量特征（包含物体与几何语义）送入 Transformer。此举绕过了VAE解码过程，大幅降低推理延迟。
*   **关键机制**：引入**记忆模块（Memory Module）**，将历史推理的潜变量作为时序条件融入扩散过程，使生成内容具备4D一致性，解决了纯视觉模型在长序列操作中的漂移问题。

### 4. 方法对比分析
*   **本质区别**：它不是简单的“以图生图”，而是将“几何重构”作为扩散模型的强约束条件，实现从几何空间到语义空间的有效映射。
*   **创新贡献**：提出无需标定的4D几何对齐策略，以及直接在扩散潜空间执行动作学习的范式，显著提升了策略在视角变化下的泛化性。
*   **适用场景**：适用于实验室或工业场景中，固定相机训练但存在多视角应用需求的机器人操作任务。

### 5. 实验分析
*   **关键结果**：在RLBench中，VistaBot的VGS指标显著领先，证明其在视角剧烈变动（±45°）下仍能维持高成功率。
*   **优势**：极强的鲁棒性，无需测试时标定，动作决策的特征分布与训练分布更贴近。
*   **局限**：在严重遮挡的情况下，扩散模型的生成质量仍会下降，限制了任务性能。

### 6. 实用指南
*   **开源情况**：作者声明代码与模型将公开。
*   **实现细节**：关键在于对VGGT进行少样本微调，以适应特定任务中的机器人本体（如夹爪）特征。训练时，重点需关注扩散模型潜空间的稳定性，建议使用PCA进行可视化分析（文中图8）。
*   **迁移建议**：可将该架构迁移至任何基于视觉的端到端动作模型（如Diffusion Policy），替换其ResNet/ViT特征提取器。

### 7. 总结
*   **核心思想**：利用几何先验引导视频生成，构建视角不变的稳健操作空间。
*   **速记版Pipeline**：
    1. 预测当前视角与训练视角的几何关系；
    2. 重投影并修复生成视觉潜变量；
    3. 融合历史记忆与几何特征；
    4. 在潜空间中完成动作决策。

**Key Findings:**

- In this paper, we propose VistaBot, a novel framework that integrates feed-forward geometric models with video diffusion models to achieve view-robust closed-loop manipulation without requiring camera calibration at test time.
- Our approach consists of three key components: 4D geometry estimation, view synthesis latent extraction, and latent action learning.
- We further introduce the View Generalization Score (VGS) as a new metric for comprehensive evaluation of cross-view generalization.
- Results show that VistaBot improves VGS by 2.79$\times$ and 2.63$\times$ over ACT and $π_0$, respectively, while also achieving high-quality novel view synthesis.
- Our contributions include a geometry-aware synthesis model, a latent action planner, a new benchmark metric, and extensive validation across diverse environments.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.21914v1)
- [arXiv](https://arxiv.org/abs/2604.21914v1)

---

<a id='2604.21911v1'></a>
## [When Prompts Override Vision: Prompt-Induced Hallucinations in LVLMs](https://arxiv.org/abs/2604.21911v1)

**Authors:** Pegah Khayatan, Jayneel Parekh, Arnaud Dapogny, Mustafa Shukor, Alasdair Newson, Matthieu Cord

**Published:** 2026-04-23

**Categories:** cs.CV, cs.AI, cs.CL, cs.LG

**Abstract:**

Despite impressive progress in capabilities of large vision-language models (LVLMs), these systems remain vulnerable to hallucinations, i.e., outputs that are not grounded in the visual input. Prior work has attributed hallucinations in LVLMs to factors such as limitations of the vision backbone or the dominance of the language component, yet the relative importance of these factors remains unclear. To resolve this ambiguity, We propose HalluScope, a benchmark to better understand the extent to which different factors induce hallucinations. Our analysis indicates that hallucinations largely stem from excessive reliance on textual priors and background knowledge, especially information introduced through textual instructions. To mitigate hallucinations induced by textual instruction priors, we propose HalluVL-DPO, a framework for fine-tuning off-the-shelf LVLMs towards more visually grounded responses. HalluVL-DPO leverages preference optimization using a curated training dataset that we construct, guiding the model to prefer grounded responses over hallucinated ones. We demonstrate that our optimized model effectively mitigates the targeted hallucination failure mode, while preserving or improving performance on other hallucination benchmarks and visual capability evaluations. To support reproducibility and further research, we will publicly release our evaluation benchmark, preference training dataset, and code at https://pegah-kh.github.io/projects/prompts-override-vision/ .

**Analysis:**

### 1. 摘要翻译
尽管大型视觉语言模型（LVLMs）取得了显著进展，但它们仍容易产生幻觉，即输出与视觉输入不符。此前研究将幻觉归因于视觉骨干网的局限性或语言组件的支配地位，但这些因素的相对重要性尚不明确。为了解决这一不确定性，我们提出了 **HalluScope**，这是一个旨在更好地理解不同因素如何诱发幻觉的基准测试。分析表明，幻觉很大程度上源于对文本先验和背景知识的过度依赖，特别是通过文本指令引入的信息。为了减轻由文本指令先验引起的幻觉，我们提出了 **HalluVL-DPO**，这是一个用于微调现成LVLMs以产生更具视觉基础（grounded）响应的框架。HalluVL-DPO利用基于样本信息量的偏好优化，结合我们构建的精选训练数据集，引导模型优先选择有据可依的响应而非幻觉响应。我们证明了优化后的模型能有效缓解针对性的幻觉失效模式，同时保持或提升在其他幻觉基准和视觉能力评估中的表现。

### 2. 方法动机分析
*   **驱动力**：现有的幻觉评估（如POPE、CHAIR）仅关注最终结果的正确性，无法区分幻觉是由视觉感知错误、语义先验还是指令诱导产生的。
*   **现有方法痛点**：模型往往“听从”指令中的暗示而非观察图像。即当指令 presuppose（预设）某物体存在时，模型即使视觉识别正确也会产生幻觉。
*   **研究假设**：LVLMs的幻觉本质上是一个**模态对齐问题**，可以通过引入包含误导性指令的偏好数据集，并结合样本加权的DPO优化，强制模型增强视觉基础（Grounding）。

### 3. 方法设计详解
*   **HalluScope基准构建**：
    1.  **样本多样化**：利用K-Center Greedy算法从COCO数据集中提取语义多样性样本。
    2.  **两阶段验证**：结合Grounding-DINO（检测）与Qwen2-VL（验证）组成闭环，确保物体“存在”或“不存在”的标签准确可靠。
    3.  **对抗性挖掘**：构建对象共现图，通过PMI（点互信息）寻找“共现概率高但视觉缺失”的物体，作为对抗性样本。
    4.  **问题生成**：生成不同类型的问题（识别、属性预设），以刻画模型对先验的依赖程度。
*   **HalluVL-DPO框架**：
    1.  **加权DPO**：引入语义差距度量（$w$），通过LLM评价偏好对的对比度，对“明显对比强”的样本给予更高权重，避免模型在简单样本上过度拟合。
    2.  **Unilateral/Contrastive提示增强**：利用“单侧”或“对比”提示生成偏好数据，配合“模型辅助的答案反转”（Model-Assisted Answer Inversion），通过LLM诱导模型生成具有细微错误属性的伪造答案，以此强制模型区分正确/错误的属性细节。

### 4. 方法对比分析
*   **本质区别**：不依赖已有的昂贵标注，而是通过LLM自动构建对抗性数据集，并引入了“样本信息量加权”机制，针对性解决指令诱导幻觉。
*   **创新贡献**：首次从定量角度解耦了幻觉的成因，并提供了一个可扩展的、自动化的数据增强流程。
*   **适用场景**：适用于任何基于Instruction Tuning的LVLM架构，尤其是在需要高可靠性、防止AI产生“指令追随导致的幻觉”的垂直领域。

### 5. 实验分析
*   **验证方法**：在HalluScope基准及现有主流基准（POPE, HallusionBench, MME等）上对比了LLaVA和Qwen2-VL的优化前后表现。
*   **关键结论**：在ADp（对抗性预设）指标上，LLaVA的性能从约5%提升至超过80%。证明了通过合成高质量偏好数据能有效纠正指令偏见。
*   **优势**：显著提升模型拒答错误预设的能力，且不会损失通用视觉任务能力。
*   **局限**：数据生成过程对基础模型的提示词遵循能力有依赖；embedding距离作为语义对比指标有时不够精准。

### 6. 实用指南
*   **开源情况**：官方将开源基准测试、代码及微调后的检查点（见项目主页）。
*   **实现细节**：$\beta$参数设为0.1；加权函数基于LLM对文本对比度的评分，建议在数据处理阶段预计算权重。
*   **迁移建议**：该方法本质是DPO的一种变体，可轻松迁移至任意支持DPO的Vision-Language架构，通过修改提示语生成策略即可适配特定垂直任务。

### 7. 总结
*   **核心思想**：通过解耦幻觉成因，利用加权对比偏好优化强行重塑模态对齐。
*   **速记版Pipeline**：
    1. 自动化挖掘视觉缺失但语义相关的对抗对象；
    2. 构造含指令暗示的偏好样本对；
    3. LLM评价样本区分度并赋予权重；
    4. 采用加权DPO微调模型。

**Key Findings:**

- To resolve this ambiguity, We propose HalluScope, a benchmark to better understand the extent to which different factors induce hallucinations.
- To mitigate hallucinations induced by textual instruction priors, we propose HalluVL-DPO, a framework for fine-tuning off-the-shelf LVLMs towards more visually grounded responses.
- We demonstrate that our optimized model effectively mitigates the targeted hallucination failure mode, while preserving or improving performance on other hallucination benchmarks and visual capability evaluations.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.21911v1)
- [arXiv](https://arxiv.org/abs/2604.21911v1)

---

<a id='2604.21776v1'></a>
## [Reshoot-Anything: A Self-Supervised Model for In-the-Wild Video Reshooting](https://arxiv.org/abs/2604.21776v1)

**Authors:** Avinash Paliwal, Adithya Iyer, Shivin Yadav, Muhammad Ali Afridi, Midhun Harikumar

**Published:** 2026-04-23

**Categories:** cs.CV

**Abstract:**

Precise camera control for reshooting dynamic videos is bottlenecked by the severe scarcity of paired multi-view data for non-rigid scenes. We overcome this limitation with a highly scalable self-supervised framework capable of leveraging internet-scale monocular videos. Our core contribution is the generation of pseudo multi-view training triplets, consisting of a source video, a geometric anchor, and a target video. We achieve this by extracting distinct smooth random-walk crop trajectories from a single input video to serve as the source and target views. The anchor is synthetically generated by forward-warping the first frame of the source with a dense tracking field, which effectively simulates the distorted point-cloud inputs expected at inference. Because our independent cropping strategy introduces spatial misalignment and artificial occlusions, the model cannot simply copy information from the current source frame. Instead, it is forced to implicitly learn 4D spatiotemporal structures by actively routing and re-projecting missing high-fidelity textures across distinct times and viewpoints from the source video to reconstruct the target. At inference, our minimally adapted diffusion transformer utilizes a 4D point-cloud derived anchor to achieve state-of-the-art temporal consistency, robust camera control, and high-fidelity novel view synthesis on complex dynamic scenes.

**Analysis:**

## 1. 摘要翻译
精准的摄像机控制是视频重拍的关键，但由于非刚性场景缺乏成对的多视图数据，导致该任务严重受阻。为了克服这一限制，我们提出了一种高度可扩展的自监督框架，能够利用互联网规模的单目视频进行训练。我们的核心贡献是生成伪多视图训练三元组，包含源视频、几何锚点和目标视频。通过从单个输入视频中提取不同的平滑随机游走裁剪轨迹作为源视图和目标视图，我们实现了这一目标。锚点是通过利用密集跟踪场对源视频的第一帧进行前向扭曲生成的，这有效地模拟了推理时所需的失真点云输入。由于这种独立的裁剪策略引入了空间失真和人工遮挡，模型不能简单地复制当前源帧的信息，而是被迫通过在源视频的不同时间和视点间主动路由和重投影丢失的高保真纹理来隐式学习4D时空结构，从而重建目标视频。在推理阶段，我们微调的扩散Transformer利用4D点云导出的锚点，实现了最先进的时间一致性、鲁棒的摄像机控制和复杂动态场景下的高保真新视图合成。

## 2. 方法动机分析
*   **驱动力**：旨在解决非刚性动态场景下视频重拍严重依赖稀缺成对多视图数据的问题，寻求利用海量单目视频实现高质量视频重拍。
*   **现有方法痛点**：纯合成数据训练存在合成到真实的泛化鸿沟（容易产生人工伪影）；仅依赖锚点（Anchor-only）的方法易受3D/4D重建误差影响且导致伪影传播；现有的显式4D模型计算昂贵且质量受限。
*   **研究假设**：通过引入一种自监督的“空间瓶颈”机制，强制模型在面对遮挡和空间失真时，学会从不同时间上下文的源视频中提取和路由纹理，从而无需显式3D标注即可隐式学习4D时空结构。

## 3. 方法设计详解
*   **流程总结**：
    1.  **数据采集与预处理**：对单目视频 $V$ 进行两次独立的平滑随机游走裁剪，得到源视频 $V_s$ 和目标视频 $V_t$。
    2.  **伪锚点生成**：利用AllTracker [11]得到密集跟踪场，结合裁剪偏移，通过Softmax Splatting [24]将 $V_s[0]$ 前向扭曲，生成几何指导锚点 $V_a$。此过程引入荧光粉背景和3D噪声注入，模拟推理时的真实失真。
    3.  **模型训练**：基于Wan 2.2-14B DiT。将 $V_s$ 和 $V_a$ 编码为隐空间，作为Conditioning输入，通过**Offset RoPE**解耦时空位置。
    4.  **损失函数**：除常规Diffusion损失外，引入辅助的**Source Token Reconstruction Loss**，确保源视频的高保真特征能够被模型有效路由。
*   **模型结构**：采用DiT架构，将 $V_s$ 和 $V_a$ 视为Token并行输入主自注意力机制，而非传统的交叉注意力。
*   **算法解释**：核心创新在于“信息擦除与重路由”。通过在训练中强制模型处理扭曲的锚点，模型被迫寻找源视频中的正确纹理来填充被遮挡或被裁剪的区域，将传统的重建任务转化为一个动态的、基于Attention的特征检索与拼接过程。

## 4. 方法对比分析
*   **本质区别**：不依赖于预训练的显式3D/4D模型，而是通过数据增强构造出一个“有缺陷”的训练任务，让模型在解决该任务的过程中“进化”出对4D结构的隐式理解。
*   **创新贡献**：
    1.  领域无关的自监督数据 pipeline。
    2.  Offset RoPE与Token拼接架构，有效解耦了源纹理提取与锚点运动控制。
    3.  通过混合15%合成数据提升极限视角下的几何稳健性。
*   **适用场景**：高动态、非刚性场景的视频重新拍摄与轨迹调整。

## 5. 实验分析（精简版）
*   **验证方法**：在Opensora-mixkit数据集上，对比ReCamMaster、EX-4D及TrajectoryCrafter，评估VBench质量、时间一致性及摄像机控制精度。
*   **关键结果**：在VBench的多项指标上均达到SOTA，特别是在复杂动作（如摄像机抖动）下，表现出极佳的纹理保留能力。
*   **优势**：极强的泛化性，高质量纹理保留，无需昂贵的3D数据。
*   **局限**：推理速度受限于Token序列长度；当目标轨迹完全超出源视频空间范围时，锚点失效。

## 6. 实用指南
*   **实现细节**：关键超参数包括 Offset RoPE 的位移常数50，源重建损失系数 $\alpha=0.1$，以及15%的合成数据混合比例。
*   **迁移可能**：可直接迁移至需要“源-目标”时空对齐的视频编辑任务。
*   **建议**：若要复现，重点在于实现稳定可靠的AllTracker轨迹提取，这是伪多视图数据生成的基础。

## 7. 总结
*   **核心思想**：通过构造伪多视图训练数据，强制模型在自监督过程中隐式习得4D时空路由能力。
*   **速记版pipeline**：1. 从视频随机裁剪出源和目标；2. 基于跟踪轨迹合成扭曲锚点；3. 将锚点和源编码为Token并输入DiT；4. 使用额外损失函数强迫模型在源视频中“搜索并拼接”纹理。

**Key Findings:**

- Our core contribution is the generation of pseudo multi-view training triplets, consisting of a source video, a geometric anchor, and a target video.
- We achieve this by extracting distinct smooth random-walk crop trajectories from a single input video to serve as the source and target views.
- At inference, our minimally adapted diffusion transformer utilizes a 4D point-cloud derived anchor to achieve state-of-the-art temporal consistency, robust camera control, and high-fidelity novel view synthesis on complex dynamic scenes.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.21776v1)
- [arXiv](https://arxiv.org/abs/2604.21776v1)

---

