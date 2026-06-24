time: 20260624

# Arxiv Computer Vision Papers - 2026-06-24

## Executive Summary

# 每日 Arxiv 计算机视觉论文执行摘要（2026-06-23）

## 一、主要主题与趋势

本期论文覆盖机器人操控、自动驾驶、3D视觉、高效视觉Transformer、多模态大模型等方向，呈现以下趋势：

- **机器人学习与操控**成为最热主题（4/10篇），聚焦于价值学习、模仿学习、模拟器辅助精调，强调鲁棒性与可解释性。
- **自动驾驶感知与推理**持续活跃，涉及鸟瞰图融合、视觉语言基础模型、异常检测等。
- **模型轻量化与高效性**是跨子领域的共同追求，如视觉Transformer的普适化、SLAM的内存优化。
- **多模态大模型与几何推理**的结合开始出现，尝试将结构化推理注入大模型中。

## 二、特别重要的创新论文

1. **《UniDrive》**（第6篇）——提出统一视觉语言与基础框架用于自动驾驶风险理解，兼具可解释性与鲁棒性，对安全关键系统意义重大。
2. **《TuringViT》**（第1篇）——致力于将SOTA ViT模型普及化，可能降低顶尖视觉Transformer的使用门槛，对资源受限场景影响深远。
3. **《World Value Models for Robotic Manipulation》**（第7篇）——引入“世界价值模型”概念，将价值函数与物理世界模型结合，为机器人操作提供新范式。

## 三、新兴研究方向与技术

- **推理时模拟器协同精调**（第9篇《Enabling Robust Cloth Manipulation》）——在推理阶段利用物理模拟器修正策略，提升操作鲁棒性，打破了“训练后固定”的传统。
- **重试监督的价值学习**（第8篇《Beyond Monotonic Progress》）——打破单调进步假设，允许失败后重试学习，更贴近真实机器人训练环境。
- **视觉链式推理**（第10篇《PointVG-R》）——将CoT思想引入多模态大模型的几何定位，显著提升空间指向精度。
- **渲染面积感知剪枝**（第4篇《Pocket-SLAM》）——针对3D高斯泼溅SLAM的内存瓶颈提出智能剪枝策略，推动实时SLAM向轻量化发展。

## 四、建议精读论文

| 优先级 | 论文 | 理由 |
|--------|------|------|
| ★★★★★ | **UniDrive** | 自动驾驶领域结合VLM与基础推理的标杆工作，实用价值高 |
| ★★★★★ | **World Value Models** | 理论创新强，可能影响机器人学习的范式 |
| ★★★★ | **TuringViT** | 对ViT推广有实际意义，适合应用研究者 |
| ★★★★ | **InSight** | 自引导技能获取与可控VLA的结合，值得关注 |
| ★★★ | **Pocket-SLAM** | 内存优化思路新颖，适合SLAM领域从业者 |

此外，**《PointVG-R》** 对MLLM几何推理感兴趣者建议阅读；**《Retry-Supervised Value Learning》** 适合研究模仿学习的学者。

---

*摘要生成时间：2026-06-23 | 共计10篇论文 | 核心主题：机器人学习、自动驾驶、模型高效化、多模态推理*

---

## Table of Contents

1. [TuringViT: Making SOTA Vision Transformers Accessible to All](#2606.24253v1)
2. [InSight: Self-Guided Skill Acquisition via Steerable VLAs](#2606.24884v1)
3. [DDStereo: Efficient Dual Decoder Transformers for Stereo 3D Road Anomaly Detection](#2606.24805v1)
4. [Pocket-SLAM: Rendering-Area-Aware Pruning for Memory-Efficient 3DGS-SLAM](#2606.24796v1)
5. [AerialFusionMapNet: Online HD Map Construction with Aerial-Onboard BEV Fusion](#2606.24784v1)
6. [UniDrive: A Unified Vision-Language and Grounding Framework for Interpretable Risk Understanding in Autonomous Driving](#2606.24759v1)
7. [World Value Models for Robotic Manipulation](#2606.24742v1)
8. [Beyond Monotonic Progress: Retry-Supervised Value Learning for Robot Imitation](#2606.24633v1)
9. [Enabling Robust Cloth Manipulation via Inference-Time Simulator-in-the-Loop Refinement](#2606.24552v1)
10. [PointVG-R: Internalizing Geometric Reasoning in MLLMs for Precise Pointing Localization via Visual Chain of Thought](#2606.24539v1)

---

## Papers

<a id='2606.24253v1'></a>
## [TuringViT: Making SOTA Vision Transformers Accessible to All](https://arxiv.org/abs/2606.24253v1)

**Authors:** Qiman Wu, Hanlin Chen, Lyujie Chen, Rui Xin, Jianlei Zheng, Mingyuan Wang, Jiahui Hu, Da Zhu, Yuecheng Ma, Yuhua Wei, Yizhao Wang, Hua Zhou, Yuheng Zhang, Anhua Liu, Shaman Tang, Yue He, Pengfei Diao, Shuang Su, Haotong Xin, Weichao Huang, Hang Zhang, Xianming Liu

**Published:** 2026-06-23

**Categories:** cs.CV

**Abstract:**

Modern VLMs and VLA systems commonly adopt off-the-shelf ViTs such as SigLIP2 as visual encoders, but diverse downstream requirements in latency, temporal modeling, and VLM integration often call for customized SOTA-level ViTs. Training such encoders remains beyond the reach of much of the community, as it requires massive image-text data, while standard softmax attention makes high-resolution or dynamic-resolution pretraining prohibitively costly and often forces low-resolution pretraining followed by post-hoc adaptation. TuringViT addresses these challenges with three key designs: Turing Linear Attention (TLA) for efficient sequence modeling, VISTA-Curation to construct supervision-rich image-video training data, and native dynamic-resolution pretraining that supports flexible inputs from the start and transfers seamlessly to downstream VLMs. As a result, TuringViT outperforms leading open-source ViT baselines with only 10% of the data, achieves stronger downstream VLM performance, and delivers substantially better latency scaling on high-resolution inputs. Our scaling-law analysis further shows that TuringViT continues to improve predictably with curated data scale, far from saturation. Its fast adaptation, hardware-friendly design, and efficient deployment have made it a unified visual foundation across XPeng's AI systems. More broadly, TuringViT provides a reproducible pipeline that dramatically lowers the cost for the community to train, customize, and deploy SOTA-level ViTs, moving toward making such Vision Transformers accessible to all.

**Analysis:**

这是一份关于 **TuringViT** 的深度方法分析报告。

### 1. 摘要翻译
现代VLM和VLA系统通常采用现成的ViT作为视觉编码器，但对延迟、时序建模和VLM集成的多样化需求往往要求定制化的SOTA级ViT。此类编码器的训练因需要海量多模态数据，且标准Softmax注意力机制在高分辨率或动态分辨率预训练中极其昂贵，常被迫采取“低分辨率预训练+后期适应”的模式，这使训练成本高昂且难以普及。
**TuringViT** 通过三项关键设计解决了这些挑战：(1) 用于高效序列建模的 **Turing Linear Attention (TLA)**；(2) 构建富监督图像-视频数据的 **VISTA-Curation**；(3) 支持灵活输入的原生动态分辨率预训练。结果显示，TuringViT仅用10%的数据即可超越主流开源基线，显著提升下游VLM性能，并在高分辨率输入下实现更好的延迟缩放。其快速适配、硬件友好及高效部署的特性使其成为XPeng AI系统的统一视觉底座。

### 2. 方法动机分析
*   **驱动力**：在受控预算下，实现一种既高效又能处理高分辨率、多模态任务的SOTA级视觉编码器。
*   **痛点**：
    *   **计算瓶颈**：标准Softmax注意力机制在长序列（高分辨率）下呈现二次复杂度。
    *   **数据效率**：大规模原始web数据存在噪声、对齐弱和冗余，导致数据缩放收益边际递减。
    *   **适配问题**：固定分辨率预训练与下游VLM动态输入需求存在严重的“分辨率不匹配”。
*   **研究假设**：通过线性注意力和强监督数据重构，可以在极小的数据规模下，通过硬件友好的架构实现更强的表示能力和更高的计算效率。

### 3. 方法设计详解
*   **Turing Linear Attention (TLA)**：核心架构创新。用线性复杂度算子替换大部分Softmax注意力，通过 $N\sqrt{d}$ 归一化提升稳定性，并引入一个**输入相关输出门（Input-dependent output gate）**，利用Sigmoid门控机制在全局聚合中保留高频细节，弥补线性注意力易平滑细节的不足。结构上采用[TLA*5, MHA*1]的重复块设计，实现高效全局聚合与局部交互的平衡。
*   **VISTA-Curation**：数据侧创新。
    *   **图像 curation**：通过多模型重标注+相对得分（基于s-CLIPLoss）+Shannon熵排序，选出信息熵高、语义密集的“金标”数据。
    *   **视频 curation**：通过光流和语义相似度过滤冗余帧，融合帧级与全局特征，将原始视频转化为具有时空一致性的高质量多模态对。
*   **原生动态分辨率训练**：放弃固定尺寸，训练初期限制最大尺寸为512，后期逐步放开，直接学习变分辨率输入，避免后期微调导致的分布偏移。

### 4. 方法对比分析
*   **本质区别**：从“堆砌数据与算力”转向“架构高效化与数据精细化”。TuringViT通过混合线性注意力实现了近线性缩放，且通过原生动态分辨率解决了空间推理难题。
*   **创新贡献**：设计了线性注意力算子TLA，解决了高分辨率视觉推理的算力瓶颈；提出了相对打分的VISTA数据引擎，大幅提升数据利用率。
*   **适用场景**：实时感知要求高、分辨率动态变化的VLA（自动驾驶）、机器人及多模态大模型底座。

### 5. 实验分析（精简版）
*   **验证方法**：在六大零样本分类基准、两大检索基准及密集预测任务（深度、分割、跟踪）上进行验证。
*   **结论**：TuringViT-24L 在仅使用10%预训练数据的情况下，平均零样本分类性能优于SigLIP2-L，且在处理超长输入时的推理延迟远低于基线。
*   **局限**：在某些严重依赖渲染风格和抽象理解的特殊数据集（如ObjectNet、Sketch）上，由于预训练数据规模较小，仍有提升空间。

### 6. 实用指南
*   **开源与部署**：官方网址：[https://turingvit.github.io](https://turingvit.github.io)。其硬件友好设计对计算资源紧缺场景极具吸引力。
*   **实现细节**：在实现TLA时，需注意引入SiLU kernel函数以保证数值稳定性；在数据过滤中，必须包含相对打分策略而非仅依赖绝对余弦相似度。
*   **迁移迁移**：方法不仅限于视觉编码器，其TLA算子和相对打分的数据CURATION策略可直接迁移至任何对序列长度敏感的 Transformer 架构中。

### 7. 总结
*   **核心思想**：通过线性注意力算子与精细化数据挖掘，实现高效率与高性能的统一。
*   **速记版Pipeline**：
    1. 使用TLA替换标准注意力，并每6层插入一层MHA以保留细节。
    2. 使用多模型重标注和基于熵的排序筛选高质量图像文本对。
    3. 视频数据通过光流滤波取关键帧并融合时空语义。
    4. 采用动态分辨率训练策略，由低向高递进学习。

**Key Findings:**

- TuringViT addresses these challenges with three key designs: Turing Linear Attention (TLA) for efficient sequence modeling, VISTA-Curation to construct supervision-rich image-video training data, and native dynamic-resolution pretraining that supports flexible inputs from the start and transfers seamlessly to downstream VLMs. As a result, TuringViT outperforms leading open-source ViT baselines with only 10% of the data, achieves stronger downstream VLM performance, and delivers substantially better latency scaling on high-resolution inputs.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24253v1)
- [arXiv](https://arxiv.org/abs/2606.24253v1)

---

<a id='2606.24884v1'></a>
## [InSight: Self-Guided Skill Acquisition via Steerable VLAs](https://arxiv.org/abs/2606.24884v1)

**Authors:** Maggie Wang, Lars Osterberg, Stephen Tian, Ola Shorinwa, Jiajun Wu, Mac Schwager

**Published:** 2026-06-23

**Categories:** cs.RO, cs.AI, cs.LG

**Abstract:**

Vision-language-action (VLA) models can learn manipulation skills from demonstrations, but their capabilities are bounded by the skills in the training data. We present InSight, a framework that unlocks autonomous skill acquisition by rendering VLAs steerable at the primitive-action level (e.g., "move gripper to the bowl", "lift upward", "pour the bottle"). InSight consists of two primary stages: (1) an automated segmentation pipeline that partitions demonstrations into labeled primitives via VLM plan decomposition and end-effector poses to enable VLA primitive steerability, and (2) a VLM-guided data flywheel that identifies missing primitives required to accomplish a novel task, autonomously attempts demonstrations of the missing primitives with VLM-proposed low-level control, and automatically labels, stores, and integrates successful demonstrations into the VLA training set. We evaluate InSight across simulation and real-world manipulation tasks, including block flipping, drawer closing, sweeping, twisting, and pouring, without any human demonstrations of these target skills. Once learned, these primitives can be composed to execute novel, long-horizon tasks without additional human demonstrations. Our findings demonstrate that primitive steerability provides a practical foundation for continual skill acquisition in VLA policies. Project website: https://insight-vla.github.io.

**Analysis:**

### 1. 摘要翻译
视觉语言动作（VLA）模型能从演示中学习操作技能，但其能力受限于训练数据中的已有技能。我们提出了 **INSIGHT**，这是一个通过使 VLA 在基元动作（primitive-action）层面（如“将抓手移动到碗上方”、“向上提起”、“倾斜瓶子”）具有可操纵性，从而解锁自主技能获取的框架。INSIGHT 由两个核心阶段组成：（1）自动分割流程，通过 VLM 计划分解和末端执行器姿态，将演示分段为标签化的基元，从而实现 VLA 的基元级可操纵性；（2）VLM 引导的数据飞轮，它能识别完成新任务所需的缺失基元，通过 VLM 提议的底层控制自主尝试完成这些缺失基元，并自动标注、存储并将成功的演示整合到 VLA 训练集中。我们在模拟和真实世界操作任务中评估了 INSIGHT，包括方块翻转、抽屉关闭、清扫、扭转和倾倒，无需任何人类提供的目标技能演示。一旦习得，这些基元即可组合起来执行长序列的新任务，而无需额外的演示。我们的研究结果表明，基元级可操纵性为 VLA 策略的持续技能获取提供了切实基础。

### 2. 方法动机分析
- **驱动力**：旨在解决传统机器人学习中“收集人类演示昂贵”且“现有模型能力受限于训练数据”的问题，实现机器人的终身持续学习。
- **痛点**：现有方法将 VLM 作为规划器仅在测试时重组已知能力，无法扩展策略本身；且现有技能往往耦合在复杂指令中，导致其不可“操纵”。
- **假设**：操纵技能具有本质上的组合性，复杂任务可分解为可复用的基元，通过识别并自主获取这些缺失的基元，可以扩展模型的技能库。

### 3. 方法设计详解
INSIGHT 的核心是通过“基元可操纵性”将策略从固定集合转变为可扩展的库。
- **阶段 1：自动基元分割**：
    - 输入：人类遥操作演示数据。
    - 流程：使用 VLM 生成逻辑计划，并结合末端执行器（EE）状态（速度、姿态、位移）对视频流进行边界对齐，实现自动标注。
    - 训练：利用 LoRA 对 VLA 进行微调，每个基元作为独立 episode，并引入“进度通道（progress channel）”作为终止信号（[0, 1] 标量）。
- **阶段 2：VLM 引导的数据飞轮**：
    - **任务分解与缺口识别**：VLM 对新任务进行规划，若步骤不在当前词汇库（vocabulary）中，即标记为“基元缺口（primitive gap）”。
    - **自主参数化执行**：VLM 为缺失基元提出单轴运动参数（坐标轴+幅值），由底层控制器执行。
    - **评估与整合**：通过 VLM 视觉 Oracle 评估成功与否，将成功轨迹注入训练集，Retrain VLA，将新技能永久固化。

### 4. 方法对比分析
- **本质区别**：与 CaP-X 等仅在推理时重组已有能力的“规划”方法不同，INSIGHT 通过**Retrain（重训练）**将缺失的动作永久性地融入策略，属于“技能增量学习”。
- **创新点**：引入了“基元级 steerability”以及“VLM 引导的自主闭环数据采集系统”。

### 5. 实验分析
- **结果**：在真实世界中实现了极高的端到端成功率（例如倾倒任务 96%，长序列任务 80%），同时保留了原有技能的性能。
- **优势**：极佳的样本效率，只需少量（约 20-30 次）自主尝试即可习得新基元；具备优秀的任务组合通用性。
- **局限**：目前基元局限于单轴运动，难以处理复杂的动态交互；且环境重置仍需手动或额外依赖。

### 6. 实用指南
- **开源/复现**：项目主页 `https://insight-vla.github.io/`。复现需重点关注 VLM 对动作轨迹的精准规划以及对 Oracle 判定条件的鲁棒性设计。
- **实现关键**：
    - **Termination Signal**：训练阶段设计的“进度通道”对区分基元边界至关重要。
    - **Constraint**：VLM 对动作的“单轴约束”是保持基元通用性的核心。
- **迁移**：该方法完全可迁移至任意具备末端执行器位移控制的机器人系统。

### 7. 总结
- **核心思想**：通过自动分割与 VLM 自主纠错，实现机器人技能的增量学习。
- **速记版 pipeline**：
    1. 自动从演示中切割出通用动作片段；
    2. 使用模型对比当前能力与新任务差距；
    3. 自动生成并尝试新动作；
    4. 评估成功后自动存入模型词库，更新策略。

**Key Findings:**

- We present InSight, a framework that unlocks autonomous skill acquisition by rendering VLAs steerable at the primitive-action level (e.g., "move gripper to the bowl", "lift upward", "pour the bottle").
- InSight consists of two primary stages: (1) an automated segmentation pipeline that partitions demonstrations into labeled primitives via VLM plan decomposition and end-effector poses to enable VLA primitive steerability, and (2) a VLM-guided data flywheel that identifies missing primitives required to accomplish a novel task, autonomously attempts demonstrations of the missing primitives with VLM-proposed low-level control, and automatically labels, stores, and integrates successful demonstrations into the VLA training set.
- Once learned, these primitives can be composed to execute novel, long-horizon tasks without additional human demonstrations.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24884v1)
- [arXiv](https://arxiv.org/abs/2606.24884v1)

---

<a id='2606.24805v1'></a>
## [DDStereo: Efficient Dual Decoder Transformers for Stereo 3D Road Anomaly Detection](https://arxiv.org/abs/2606.24805v1)

**Authors:** Shiyi Mu, Zichong Gu, Zhiqi Ai, Yilin Gao, Shugong Xu

**Published:** 2026-06-23

**Categories:** cs.CV

**Abstract:**

Stereo-based 3D object detection still faces two critical safety challenges: real-time performance and open-set generalization. Existing stereo 3D methods typically achieve twice the accuracy of monocular methods but suffer from significantly lower inference speeds, making them unsuitable for real-time applications. Meanwhile, recent advances in open-world detection have introduced open-set and open-vocabulary algorithms in monocular 2D and 3D settings, yet stereo-based open-set detection remains largely unexplored. To bridge this gap, we propose DDStereo, a novel Dual-Decoder Stereo Transformer for real-time open-set 3D object detection. DDStereo features two lightweight decoder branches: one for open-set foreground 2D detection and the other for 3D attribute regression. These decoders share object-level queries to achieve unified target-level alignment. To enhance inference efficiency, we designed a compact disparity feature extractor and a streamlined decoder architecture. Experiments on public stereo 3D benchmarks demonstrate that DDStereo achieves state-of-the-art accuracy under both closed-set and open-set protocols. Notably, our method surpasses existing stereo 3D detectors in inference speed and, for the first time, achieves real-time performance comparable to monocular approaches.

**Analysis:**

### 1. 摘要翻译
基于立体视觉的3D目标检测仍面临实时性能和开放集泛化两大安全挑战。现有的立体3D方法虽然精度通常是单目方法的两倍，但推理速度显著较慢，难以满足实时应用需求。同时，尽管开放世界检测在单目2D和3D领域已有进展，但基于立体视觉的开放集检测仍未得到充分探索。为此，我们提出了DDStereo，一种用于实时开放集3D目标检测的新型双解码器立体Transformer。DDStereo具有两个轻量级解码器分支：一个用于开放集前景2D检测，另一个用于3D属性回归，两者共享对象查询以实现统一的目标级对齐。为提高推理效率，我们设计了紧凑的视差特征提取器和精简的解码器架构。实验表明，DDStereo在封闭集和开放集协议下均达到了最先进的精度。值得注意的是，该方法在推理速度上超越了现有的立体3D检测器，并首次实现了可与单目方法媲美的实时性能。

### 2. 方法动机分析
*   **驱动力**：在自动驾驶中，除了已知类别（如车辆、行人）的检测，还需要识别“未知障碍物”（Out-of-Distribution, OoD），且需满足实时性（<30ms）。
*   **现有痛点**：
    *   现有立体方法多采用重型3D代价卷或复杂锚框设计，推理延迟高。
    *   多数开放集方法依赖文本提示或复杂的多模态融合，导致计算开销过大。
    *   现有方案多为封闭集，缺乏对未知障碍物的有效发现能力。
*   **核心假设**：前景检测（物体是否存在）和分类（它是哪类）可以解耦，且利用视差图（几何特征）比利用RGB纹理更易于在不依赖类别标签的情况下定位未知目标。

### 3. 方法设计详解
*   **流程总结**：
    1.  **特征提取**：共享Backbone提取多尺度视觉特征，利用左右视图构建视差特征（Correlation Volume）。
    2.  **双解码器架构**：
        *   **前景检测分支**：基于视差特征，通过二分类判断像素是否为“前景”，实现类别无关的目标定位。
        *   **3D检测分支**：基于视觉特征，回归3D属性（尺寸、方向、深度等）。
    3.  **查询对齐（Shared Queries）**：两分支共享一组可学习的Object Queries，确保前景框与3D属性框在空间上的一致性，省去传统的NMS操作。
    4.  **深度预测与融合**：利用U形模块从视差特征中直接预测物体级深度图，通过中心采样增强定位精度。
    5.  **异常得分计算（MNPF）**：通过“最大已知概率”与“前景置信度”之差，判断该物体是否为OoD对象。

### 4. 方法对比分析
*   **本质区别**：将3D检测任务解耦为“几何驱动的前景定位”和“视觉驱动的类别回归”，利用共享查询桥接，而非传统的堆叠融合。
*   **创新点**：
    *   **双解码器结构**：实现了对未知障碍物的显式分类，无需文本输入。
    *   **视差驱动的前景定位**：完全利用几何信息定位，对材质不敏感。
    *   **MNPF算法**：提出了一种极其轻量的异常评分指标，无需额外模型即可识别OoD。

### 5. 实验分析
*   **关键结果**：在KITTI测试集上达到23.5ms的实时推理速度，同时在Moderate和Hard级别上实现了SOTA级精度。
*   **主要优势**：极高的推理效率（62.65 GFLOPs），参数量仅为传统SOTA的1/5左右，真正实现了立体视觉检测的实时化。
*   **主要局限**：对于那些在训练集中频繁作为“背景”出现的物体（如某些特定黑箱），模型容易将其误识别为背景。

### 6. 实用指南
*   **实现细节**：建议使用Adam优化器，初始学习率0.0002，在125和165 epoch进行学习率衰减。Grid Sample操作是保证精度与速度平衡的关键，避免使用复杂的跨模态对齐。
*   **迁移可能**：该解耦架构可轻易移植到其他需要实时异常检测的场景（如工业质检、机器人避障），只需替换后端分类头即可。

### 7. 总结
*   **核心思想**：通过解耦前景定位与类别回归，实现轻量级实时开放集3D检测。
*   **速记版Pipeline**：
    1. 提取左右视差特征。
    2. 前景分支定位几何物体。
    3. 3D分支回归属性类别。
    4. 共享查询实现空间对齐。
    5. 计算置信差甄别未知障碍物。

**Key Findings:**

- To bridge this gap, we propose DDStereo, a novel Dual-Decoder Stereo Transformer for real-time open-set 3D object detection.
- Experiments on public stereo 3D benchmarks demonstrate that DDStereo achieves state-of-the-art accuracy under both closed-set and open-set protocols.
- Notably, our method surpasses existing stereo 3D detectors in inference speed and, for the first time, achieves real-time performance comparable to monocular approaches.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24805v1)
- [arXiv](https://arxiv.org/abs/2606.24805v1)

---

<a id='2606.24796v1'></a>
## [Pocket-SLAM: Rendering-Area-Aware Pruning for Memory-Efficient 3DGS-SLAM](https://arxiv.org/abs/2606.24796v1)

**Authors:** Leshu Li, Jie Peng, Yang Zhao

**Published:** 2026-06-23

**Categories:** cs.CV

**Abstract:**

3D Gaussian Splatting (3DGS) has garnered significant attention in Simultaneous Localization and Mapping (SLAM) due to its advances in capturing fine-grained geometry features and synthesizing novel views. For SLAM in large-scale scenes, such as autonomous driving, 3DGS-SLAM faces a critical limitation: memory consumption increases continuously over time as Gaussian points accumulate, leading to poor memory efficiency and limiting its applicability. In this work, we propose a rendering-area-aware pruning strategy that selectively removes Gaussians based on their contribution to the effective rendering area, rather than solely relying on Gaussian-level heuristics such as opacity or gradient magnitude. This perspective directly targets the sources of memory redundancy, effectively reducing the peak memory footprint of 3DGS-SLAM during runtime. Evaluations on the EuRoC and KITTI datasets demonstrate that our method consistently outperforms existing pruning approaches in large-scale outdoor scenes, achieving over 60% memory reduction and more than 2 times FPS improvement while preserving localization and mapping accuracy. These results highlight rendering-area-aware pruning as a promising direction for scaling 3DGS-SLAM to real-world autonomous driving scenarios. Our code is publicly available at https://github.com/UMN-ZhaoLab/Pocket-SLAM.git.

**Analysis:**

### 1. 摘要翻译
3D高斯泼溅（3DGS）因其在捕捉细粒度几何特征和合成新视角方面的进展，在同步定位与建图（SLAM）领域受到极大关注。然而，对于自动驾驶等大规模场景，3DGS-SLAM 面临一个关键瓶颈：随着高斯点不断累积，内存消耗持续增长，限制了其在资源受限边缘设备上的部署。本文提出了一种“渲染区域感知”（rendering-area-aware）的修剪策略，它不再单纯依赖高斯点级别的启发式指标（如不透明度或梯度大小），而是根据高斯点对有效渲染区域的贡献程度进行选择性移除。该视角直接解决了内存冗余的来源，有效降低了 3DGS-SLAM 运行时的峰值内存占用。在 EuRoC 和 KITTI 数据集上的评估表明，我们的方法在大型户外场景中明显优于现有的修剪方案，在保持定位和建图精度的前提下，实现了超过 60% 的内存缩减和 2 倍以上的帧率提升。这些结果突显了渲染区域感知修剪在将 3DGS-SLAM 扩展至现实自动驾驶场景中的潜力。

---

### 2. 方法动机分析
- **驱动力**：解决大规模户外场景中 3DGS-SLAM 模型随时间推移内存爆炸、导致边缘设备无法运行的难题。
- **痛点**：现有修剪方法大多针对小规模室内场景，或仅关注关键帧存储，忽略了运行时峰值内存。此外，基于不透明度或梯度的修剪会导致纹理丰富区域（包含大量小高斯）被过度修剪，造成信息丢失。
- **研究假设**：高斯点对渲染图像的贡献不仅仅取决于其自身属性，还取决于其在图像平面的“有效覆盖区域”；通过结合空间分布的预算约束，可以实现内存节省与精度保持的平衡。

---

### 3. 方法设计详解
**核心 Pipeline：**
1. **渲染区域评估**：在映射阶段，通过计算每个高斯点在当前帧图像平面的贡献度 $\alpha_i(p)$，累加得到覆盖权重 $S_i$。该指标反映了该高斯点对最终图像合成的贡献。
2. **Tile-Level Budget 机制**：这是本方案的创新补充。为防止区域过度修剪，图像被划分为多个 Tile。通过追踪过程中的平均梯度 $G_k$（衡量纹理密度），为不同 Tile 分配生存预算 $B_k^{trk}$。
3. **修剪决策**：在映射阶段，根据 Tile 分配的预算，保留 $S_i$ 权重最高的一批高斯点。

**算法公式解释：**
- **Eq. 10/11 (渲染区域权重)**：计算高斯点在图像空间投影的累积权重，核心是量化该点对成像贡献的“面积感”。
- **Eq. 13 (Tile 预算分配)**：通过 `clip` 函数限制，确保任何区域（即使是低纹理区）都有最小生存额度 $B_{min}$，同时防止高纹理区占用过多内存。

---

### 4. 方法对比分析
- **本质区别**：从“局部属性驱动”转向“场景渲染贡献驱动”，并引入了“Tile 空间平衡约束”。
- **创新贡献**：
    - 引入了 rendering-area-aware 指标，更客观地反映高斯点价值。
    - 设计了 Tile-level budget 机制，解决了户外场景中“纹理稠密区”与“天空/道路区”修剪不均衡的问题。
- **适用场景**：大规模户外环境（自动驾驶、无人机）、计算资源受限的边缘嵌入式系统。

---

### 5. 实验分析（精简版）
- **验证方法**：在 EuRoC 和 KITTI 两个公开基准数据集上，对比 LSG-SLAM 等主流算法。
- **关键结果**：在 KITTI 数据集上，相比原始方法内存降低 65.7%，FPS 提升 2.9 倍，且未损害定位精度（ATE 保持一致）。
- **优势**：极高的内存压缩比，在纹理稀疏与稠密区域之间达到了很好的鲁棒性。
- **局限**：对极小尺度场景的性能优势可能不如大规模复杂场景明显，且依赖于 Tile 划分的超参数调优。

---

### 6. 实用指南
- **开源情况**：已开源，代码库地址：[https://github.com/UMN-ZhaoLab/Pocket-SLAM](https://github.com/UMN-ZhaoLab/Pocket-SLAM)
- **实现细节**：
    - `N_tar` (目标高斯点总数) 设为 `0.4 * N_init`。
    - `B_min` = 5，`B_max` = 200，这些参数对平衡重建质量至关重要。
- **迁移可能**：该修剪逻辑可直接嵌入任何基于 3DGS 的渲染/SLAM 框架，只需替换其剪枝模块即可。

---

### 7. 总结
- **核心思想**：基于渲染贡献度与分块空间约束的平衡修剪。
- **速记版 Pipeline**：
    1. 统计各 Tile 梯度，确定生存预算。
    2. 计算高斯点在当前视角的覆盖贡献。
    3. 根据预算，仅保留高贡献高斯点。
    4. 执行优化并进入下一帧迭代。

**Key Findings:**

- 3D Gaussian Splatting (3DGS) has garnered significant attention in Simultaneous Localization and Mapping (SLAM) due to its advances in capturing fine-grained geometry features and synthesizing novel views.
- In this work, we propose a rendering-area-aware pruning strategy that selectively removes Gaussians based on their contribution to the effective rendering area, rather than solely relying on Gaussian-level heuristics such as opacity or gradient magnitude.
- Evaluations on the EuRoC and KITTI datasets demonstrate that our method consistently outperforms existing pruning approaches in large-scale outdoor scenes, achieving over 60% memory reduction and more than 2 times FPS improvement while preserving localization and mapping accuracy.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24796v1)
- [arXiv](https://arxiv.org/abs/2606.24796v1)

---

<a id='2606.24784v1'></a>
## [AerialFusionMapNet: Online HD Map Construction with Aerial-Onboard BEV Fusion](https://arxiv.org/abs/2606.24784v1)

**Authors:** Daniel Lengerer, Mathias Pechinger, Klaus Bogenberger, Carsten Markgraf

**Published:** 2026-06-23

**Categories:** cs.CV

**Abstract:**

High-resolution aerial imagery has recently emerged as a complementary modality for automated driving perception and has shown potential to improve birds-eye-view (BEV) scene understanding when fused with onboard sensors. Prior work demonstrated performance gains for online high-definition (HD) map construction through aerial-onboard fusion; however, conventional end-to-end fusion does not fully exploit the structural information contained in aerial representations. In this work, we introduce AerialFusionMapNet, a fusion-based mapping framework with a structured two-stage training strategy that explicitly enhances the contribution of aerial features within a unified pipeline. The proposed training scheme enables more effective integration of structural aerial priors. On the nuScenes geographic split, AerialFusionMapNet achieves up to 54.7 mAP, improving over prior aerial-onboard fusion baselines from 48.8 mAP by +5.9 absolute and +12.1% relative. The results suggest that structured training design, rather than increased architectural complexity, plays a more decisive role in unlocking the full potential of aerial imagery for online HD map construction. Code and trained models are available at https://github.com/DriverlessMobility/AerialFusionMapNet.

**Analysis:**

以下是针对 *AerialFusionMapNet* 论文的深度技术分析：

### 1. 摘要翻译
高分辨率航空影像已成为自动驾驶感知中一种重要的辅助模态，并展现出与车载传感器融合以改善鸟瞰图（BEV）场景理解的潜力。尽管现有工作证明了航空-车载融合在在线高清地图构建中的性能增益，但传统的端到端融合未能充分利用航空表示中的结构信息。本文提出了 *AerialFusionMapNet*，这是一个融合框架，通过结构化的两阶段训练策略显式增强了航空特征在统一流水线中的贡献。该策略实现了结构化航空先验的有效整合。在 nuScenes 地理分割基准上，*AerialFusionMapNet* 达到 54.7 mAP，较现有的航空-车载融合基线（48.8 mAP）提升了 5.9 个绝对值和 12.1% 的相对值。结果表明，结构化的训练设计比增加架构复杂度对于释放航空影像的潜力更具决定性作用。

### 2. 方法动机分析
*   **核心驱动力**：解决端到端训练中“航空信息利用不足”的问题，通过引入先验知识提升特征对齐效果。
*   **现有痛点**：现有工作大多采用简单的端到端训练，航空特征与车载 BEV 特征之间存在表征差异，导致“融合难、利用浅”。
*   **核心假设**：航空特征应当作为一种“强结构先验”，通过分阶段学习和跨视角对齐（CVS），而非简单地在输入端叠加，能产生更好的融合表示。

### 3. 方法设计详解
*   **流程总结**：
    1.  **Stage 1：航空独立预训练**：仅使用航空图训练编码器。核心技术是 **SCR (Scenario-Consistent Rotation)**，通过对场景进行统一旋转增强，迫使编码器学到几何一致的地图结构，而不依赖于特定的 ego-vehicle 位置。
    2.  **Stage 2：联合训练与耦合**：固定 Stage 1 训练好的航空编码器。在车载 BEV 特征和航空特征之间引入 **CVS (Cross-View Supervision)**。
*   **关键技术**：
    *   **SCR**：在数据预处理阶段对航空图和对应的 BEV 标签进行同向旋转，提升模型旋转不变性。
    *   **CVS 与 仿射对齐**：使用均方误差（MSE）约束车载特征向航空特征对齐。引入了一个**轻量级仿射模块**（可学习的通道缩放与偏置），解决两个分支在统计分布上的不匹配，防止教师（航空）带来的噪声干扰学生（车载）。

### 4. 方法对比分析
*   **本质区别**：从传统的“直接融合输入”转变为“特征空间引导与对齐”，强调航空特征作为稳定参考的作用。
*   **创新点**：提出了分阶段训练策略，通过冻结并预训练的航空编码器提供“纯净”的结构参照，并通过 CVS 进行显式特征耦合。
*   **适用场景**：适用于城市驾驶场景，特别是有遮挡、视野受限、且有离线航空地图支撑的环境。

### 5. 实验分析
*   **验证方法**：在 nuScenes 地理分割基准（更具挑战性的跨区域泛化）上测试。
*   **关键结论**：编码器容量（Params）与性能不成正比，结构化的训练策略（Stage 1+2）比单纯堆叠参数更能提升性能。
*   **优势**：在无显式深度对齐情况下具有较强的鲁棒性；对空间错位（0.6m 以内）表现稳定。
*   **局限**：对“训练集与验证集地理覆盖重叠”极度敏感，地理重叠会导致性能虚高（在非重叠区域性能下降明显）。

### 6. 实用指南
*   **开源地址**：[https://github.com/DriverlessMobility/AerialFusionMapNet](https://github.com/DriverlessMobility/AerialFusionMapNet)
*   **实现细节**：
    *   使用 AdamW 优化器，学习率 5e-4，余弦退火策略。
    *   CVS 的 $\lambda_{CVS}$ 超参数（60-70）需针对不同分辨率进行微调。
*   **迁移建议**：该思路（分阶段预训练+跨模态对齐）可直接迁移至任何存在“强弱模态”对比（如 LiDAR-Camera 融合）的多模态感知任务中。

### 7. 总结
*   **核心思想**：通过先验引导的阶梯式学习取代盲目的端到端融合。
*   **速记版 Pipeline**：
    1. 预训练航空编码器（引入统一旋转增强）；
    2. 固定航空分支作为参考；
    3. 加入跨视角特征监督，强迫车载分支学习空间结构；
    4. 联合微调最终预测头。

**Key Findings:**

- In this work, we introduce AerialFusionMapNet, a fusion-based mapping framework with a structured two-stage training strategy that explicitly enhances the contribution of aerial features within a unified pipeline.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24784v1)
- [arXiv](https://arxiv.org/abs/2606.24784v1)

---

<a id='2606.24759v1'></a>
## [UniDrive: A Unified Vision-Language and Grounding Framework for Interpretable Risk Understanding in Autonomous Driving](https://arxiv.org/abs/2606.24759v1)

**Authors:** Xiaowei Gao, Pengxiang Li, Yitai Cheng, Ruihan Xu, James Haworth, Stephen Law, Yun Ye

**Published:** 2026-06-23

**Categories:** cs.CV, cs.AI

**Abstract:**

Recent multimodal large language models (MLLMs) have shown strong potential for autonomous driving scene understanding, yet existing methods still face a fundamental trade-off between temporal reasoning and spatial precision. Models that rely on single-frame or low-resolution inputs often miss small, distant, or partially occluded hazards, while language-centric driving models frequently provide limited grounded evidence for their explanations. To address this gap, we propose UniDrive, a unified visual-language and grounding framework for interpretable risk understanding in autonomous driving. UniDrive combines a temporal reasoning branch that models scene dynamics from multi-frame visual input with a high-resolution perception branch that preserves fine-grained spatial details from the latest frame. The two branches are integrated through a gated cross-attention fusion module, enabling dynamic context to be aligned with precise spatial evidence. Based on the fused representation, UniDrive jointly generates natural-language risk descriptions and grounded bounding-box outputs for risk objects. Experiments on the DRAMA-Reasoning benchmark show that UniDrive outperforms representative image-based and video-based baselines in both captioning and risk-object grounding. In particular, UniDrive achieves the best overall performance on the validation split and demonstrates clear advantages in small-object localization, zero-shot generalization to NuScenes and BDD100K, and human-rated interpretability and trustworthiness. These results suggest that explicitly combining temporal semantics and high-resolution perception provides a stronger foundation for interpretable and safety-oriented autonomous driving systems. The code is available at https://github.com/pixeli99/unidrive-dev.

**Analysis:**

### 1. 摘要翻译
近期，多模态大语言模型（MLLM）在自动驾驶场景理解领域展现出巨大潜力，但现有方法在处理时序推理与空间精度时存在权衡。依赖单帧或低分辨率输入的方法常忽略细小或遮挡的风险物体，而以语言为中心的方法又缺乏对解释的细粒度视觉证据支持。为解决此问题，我们提出 **UniDrive**，一个用于自动驾驶中可解释风险理解的统一视觉-语言与基础（Grounding）框架。该框架结合了时序推理分支（建模多帧场景动态）与高分辨率感知分支（保留最新帧的精细空间细节），通过门控交叉注意力融合模块将动态上下文与空间证据对齐。UniDrive 不仅能生成自然语言风险描述，还能同步输出风险物体的精准边界框。在 DRAMA-Reasoning 基准测试及其他驾驶数据集上的实验表明，UniDrive 在风险描述与目标定位上均优于主流方法，且在小目标定位、零样本泛化能力及人类感知评价方面表现卓越。

---

### 2. 方法动机分析
*   **驱动力**：解决自动驾驶中“解释”与“定位”脱节的问题，确保风险推理既具有动态时序逻辑，又具备严谨的视觉证据。
*   **现有方法痛点**：
    1.  **时序缺失与分辨率折中**：基于单帧的方法缺失时序交互信息；基于视频的方法常因低分辨率导致对遮挡物或远方小目标定位不准。
    2.  **证据匮乏**：现有模型生成的解释往往缺乏对应的空间边界框，导致其可解释性仅停留在语义层面，缺乏可信度。
*   **研究假设**：通过显式地融合时序上下文与高分辨率静态空间证据，可以显著提升模型在复杂场景下的风险定位精度及解释的可验证性。

---

### 3. 方法设计详解
UniDrive 的核心是双分支架构与门控融合机制：
*   **时序推理分支 (T-RB)**：输入多帧低分辨率视频，通过预训练 ViT-L/14 提取特征，经轻量级聚合器进行时序池化，捕捉场景的运动动态和意图。
*   **高分辨率感知分支 (P-B)**：针对最后一帧图像进行高分辨率处理，利用独立的 ViT 提取高保真特征图，保留关键的细微空间信息。
*   ** spatio-temporal 融合 (STF)**：这是本文的核心创新。利用 T-RB 的聚合特征作为 **Query**，将 P-B 的高分辨率特征映射为 **Key** 和 **Value**，通过“门控交叉注意力”进行计算。
    *   **门控机制**：公式 $F^{fused} = \alpha \cdot \text{Attention}(Q, K, V) + (1 - \alpha) \cdot Q$。通过可学习参数 $\alpha$ 动态平衡时序语义与空间细节，实现模型“带着目的（风险动态）去找细节（位置坐标）”。
*   **输出生成**：将融合特征喂入冻结的 LLM（Llama2-7B），将风险物体的边界框坐标直接编码为 `<box>x1, y1, x2, y2</box>` 的文本标记，实现端到端的风险描述与定位联合输出。

---

### 4. 方法对比分析
*   **本质区别**：不同于传统的“先检测后描述”或简单的特征拼接，UniDrive 实现了时序上下文与空间 evidence 的**动态交互式对齐**。
*   **创新贡献**：引入门控交叉注意力机制，利用风险事件的时序动机引导对空间细节的关注，极大增强了小目标识别率。
*   **适用场景**：极端天气（夜间、雨天）、长尾驾驶场景（遮挡、远距离小目标）、需要提供可审计决策的自动驾驶系统。

---

### 5. 实验分析（精简版）
*   **关键结果**：在 DRAMA-Reasoning 基准上，相比 Video-LLAMA，UniDrive 在 CIDEr（描述质量）和 mIoU（定位精度）指标上大幅领先，尤其在小目标定位上提升显著。
*   **优势**：强鲁棒性（在夜间/雨天环境表现出更小的性能降幅）及高水平的“可解释性”指标（人类评价中 trustworthiness 领先优势明显）。
*   **局限**：在多个风险目标并存时，模型有时会产生“风险优先级”偏差，倾向于关注视觉更显著的非关键目标。

---

### 6. 实用指南
*   **实现细节**：训练分阶段进行，T-RB 分支采用 $1 \times 10^{-4}$ 的学习率以细调预训练权重，P-B 分支采用 $4 \times 10^{-4}$ 以快速适应新任务。
*   **开源情况**：代码已开源至 [https://github.com/pixeli99/unidrive-dev](https://github.com/pixeli99/unidrive-dev)。
*   **迁移建议**：其 spatio-temporal fusion 模块可独立拆解，用于任何需要将视频流特征与高分辨率图片特征对齐的多模态任务。

---

### 7. 总结
*   **核心思想**：通过时序逻辑 Query 高分辨率视觉特征，实现风险理解的端到端语义与空间联动。
*   **速记版pipeline**：
    1.  视频流编码并聚合时序特征。
    2.  对关键帧提取高分辨空间特征。
    3.  通过门控注意力将两者对齐融合。
    4.  LLM 生成带有框坐标标签的自然语言解释。

**Key Findings:**

- To address this gap, we propose UniDrive, a unified visual-language and grounding framework for interpretable risk understanding in autonomous driving.
- Experiments on the DRAMA-Reasoning benchmark show that UniDrive outperforms representative image-based and video-based baselines in both captioning and risk-object grounding.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24759v1)
- [arXiv](https://arxiv.org/abs/2606.24759v1)

---

<a id='2606.24742v1'></a>
## [World Value Models for Robotic Manipulation](https://arxiv.org/abs/2606.24742v1)

**Authors:** Zhihao Wang, Jianxiong Li, Yu Cui, Yuan Gao, Xianyuan Zhan, Junzhi Yu, Xiao Ma

**Published:** 2026-06-23

**Categories:** cs.RO

**Abstract:**

Generalist value models play a pivotal role in scaling robotic policy learning from large-scale, mixed-quality data. Mathematically, accurate value estimation demands deep temporal understanding, requiring models to both ground the current belief using historical context and plan over future outcomes. However, most existing robotic value models are built on Vision-Language Model (VLM) backbones that are pretrained primarily on static or temporally sparse visual observations, lacking the requisite temporal modeling capabilities for value estimation. Unlike VLMs, world models naturally excel at temporal modeling and future planning, making them ideal foundations for learning generalizable value functions. Driven by this insight, we marry world models with value estimation to construct a new generalist robotic value model, World Value Model (WVM), that offers accurate task progressions to assess data quality. On standard benchmarks, WVM delivers state-of-the-art (SOTA) Value-Order Correlation (VOC) results. Complementing standard evaluation suites that contains only expert data, we further introduce Suboptimal-Value-Bench, a multi-embodiment benchmark consisting of 800 suboptimal trajectories with high-fidelity, human-labeled frame annotations. Our evaluations show that WVM maintains its SOTA performance on Suboptimal-Value-Bench, establishing its robustness in handling both expert and suboptimal data. When deployed for policy learning, WVM improves manipulation performance across various policy extraction approaches in both simulated and real-world deployment, providing robust guidance for learning from mixed-quality data.

**Analysis:**

以下是对论文《World Value Models for Robotic Manipulation》的方法论深度分析：

### 1. 摘要翻译
通用价值模型在利用大规模混合质量数据进行机器人策略学习中起着至关重要的作用。数学上，准确的价值估计需要深度的时序理解，要求模型既能通过历史上下文锚定当前信念，又能对未来结果进行规划。然而，大多数现有的机器人价值模型基于视觉语言模型（VLM）骨干，这些模型主要在静态或时序稀疏的视觉观测上预训练，缺乏价值估计所需的时序建模能力。与VLM不同，世界模型在时序建模和未来规划方面具有天然优势，是学习通用价值函数的理想基础。基于这一洞察，我们将世界模型与价值估计相结合，构建了一种新的通用机器人价值模型——世界价值模型（WVM），它通过准确的任务进度预测来评估数据质量。在标准基准测试上，WVM提供了最先进的价值顺序相关性（VOC）结果。除了仅包含专家数据的标准评估套件外，我们还引入了Suboptimal-Value-Bench，这是一个由800条带有高保真人工标注轨迹的多模态基准。评估表明，WVM在处理专家和次优数据时均保持了SOTA性能。在部署到策略学习中时，WVM在模拟和真实世界的部署中都提高了各种策略提取方法的操纵性能，为从混合质量数据中学习提供了可靠的指导。

### 2. 方法动机分析
*   **驱动力**：作者认为价值估计的本质是“任务进度预测”，而这一过程不仅需要空间感知，更依赖于对时序动态的理解与未来状态的推演。
*   **现有方法痛点**：
    1.  依赖标量监督，导致训练低效。
    2.  任务专有化严重，缺乏通用性。
    3.  基于VLM的方案对时序建模能力弱，难以处理次优轨迹（如犹豫、重试）。
*   **研究假设**：预训练视频世界模型中蕴含的“时空先验”可以通过精巧的架构迁移到价值函数学习中，从而实现比传统VLM更精准的价值度量。

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入处理**：将观测序列 $o_{t-h+1:t}$ 及其对应的未来帧和前缀帧输入视频VAE，编码为时空潜空间表示。
    2.  **双流耦合（MoT）**：采用轻量级DiT作为价值流，通过“混合Transformer（MoT）”架构，利用非对称注意力机制，让价值令牌（Value Tokens）关注视频特征，但视频流不受价值流干扰。
    3.  **分布建模**：将价值函数从标量回归转变为分布块（Distributional Chunk）预测，利用流匹配（Flow Matching）进行训练，提供比传统监督更稠密的信号。
*   **关键公式**：$p_{\psi}(\hat{v}_{t-h+1:t} | M_{\omega}(o_{t-h+1:t}, l))$，将价值估计简化为基于世界模型特征提取器的条件分布预测。
*   **训练增强**：采用“前缀随机化”防止模型依赖前缀产生虚假相关，以及“视频重放增强”来模拟次优行为，使模型学会识别非单调进度。

### 4. 方法对比分析
*   **本质区别**：从传统的“基于分类/回归的奖励模型”转向了“基于预训练世界模型的流匹配生成模型”。
*   **创新贡献**：首次将大规模预训练视频生成模型迁移到价值估计任务；提出分布式价值块学习方案；针对次优轨迹量身定制了评估基准。
*   **适用场景**：在大规模、混合质量的机器人操纵数据集上进行离线数据清洗和离线强化学习策略优化。

### 5. 实验分析
*   **关键结论**：在Suboptimal-Value-Bench上，WVM的Hesitation-RMSE显著低于基线， Retry-VOC表现极佳。
*   **主要优势**：极强的鲁棒性，特别是在处理停顿、重试等非专家行为时，相比传统VLM大幅降低了“预测漂移”。

### 6. 实用指南
*   **开源情况**：作者提供了主页（zh1hao.wang/wvm），承诺开源模型与基准。
*   **实现细节**：
    *   **架构**：基于Wan2.2-Ti2V-5B checkpoint，价值DiT参数量约0.7B。
    *   **超参数**：推荐 $p=0.5$（前缀随机化比例），$\lambda=1.0$（视频共训练权重）。
*   **迁移可能**：该架构可以轻松迁移至任何具备强大视频生成能力的基础模型上，只需替换VAE输入接口即可。

### 7. 总结
*   **核心思想**：利用世界模型时空先验，将价值评估转化为连续时序块的概率分布预测。
*   **速记版pipeline**：
    1. 视频VAE提取时空潜空间特征。
    2. 双流Transformer耦合，价值流通过非对称注意力获取时空信息。
    3. 使用流匹配算法预测价值块分布。
    4. 通过前缀随机化和数据增强实现稳健的时序评估。

**Key Findings:**

- Driven by this insight, we marry world models with value estimation to construct a new generalist robotic value model, World Value Model (WVM), that offers accurate task progressions to assess data quality.
- On standard benchmarks, WVM delivers state-of-the-art (SOTA) Value-Order Correlation (VOC) results.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24742v1)
- [arXiv](https://arxiv.org/abs/2606.24742v1)

---

<a id='2606.24633v1'></a>
## [Beyond Monotonic Progress: Retry-Supervised Value Learning for Robot Imitation](https://arxiv.org/abs/2606.24633v1)

**Authors:** Xinyao Qin, Junjie Lu, Kaixin Wang, Chuheng Zhang, Sinjae Kang, Kimin Lee, Min Xu, Bin Liang, Jun Yang, Li Zhao

**Published:** 2026-06-23

**Categories:** cs.RO

**Abstract:**

Human demonstrations for robot imitation learning often contain mistakes and corrective behaviors, such as imprecise grasps, object misalignment, unstable contact, and repeated attempts. While these segments are commonly treated as noisy or suboptimal data, they provide valuable evidence about when execution deviates from a desirable path and how task feasibility can be restored. However, existing reward and value models often rely on monotonic progress assumptions, which capture coarse task advancement but may overlook local execution errors and corrective behaviors in imperfect demonstrations. In this work, we propose ReTVL (ReTry-Supervised Value Learning), a framework for learning mistake-sensitive value functions from mixed-quality robot demonstrations by leveraging retry events as sparse supervision. ReTVL captures the local degradation-and-recovery structure around mistakes by combining global progress calibration with local pairwise preference learning induced by sparsely annotated retry keypoints. The learned value model is then used to reweight demonstration chunks for downstream behavior cloning, reducing the influence of harmful execution errors while preserving useful corrective behaviors. Experiments on real-robot manipulation tasks show that ReTVL produces more fine-grained value estimates than progress-based baselines and improves imitation learning from imperfect demonstrations.

**Analysis:**

这是一份关于《Beyond Monotonic Progress: Retry-Supervised Value Learning for Robot Imitation》(ReTVL) 的深度解析：

### 1. 摘要翻译
机器人模仿学习的演示数据常包含错误和纠正行为（如抓取不准、物体错位等）。传统方法往往假设任务进度是单调增加的，从而忽略了这些包含宝贵纠正信息的非单调片段。本文提出 **ReTVL (Retry-Supervised Value Learning)**，一种利用“重试事件”作为稀疏监督信号的价值学习框架。ReTVL 通过结合全局进度校准与重试关键点诱导的局部成对偏好学习，捕捉错误前后的降级与恢复结构。实验表明，ReTVL 学习到的价值函数能更精准地识别不良片段并保留有用的纠正行为，从而显著提升下游行为克隆策略的成功率。

### 2. 方法动机分析
*   **驱动力**：利用机器人演示中常见的“失败-纠正”行为，将原本被视为噪音的数据转化为高价值的局部监督信号。
*   **痛点**：现有方法（如基于进度回归的模型）强制假设价值随时间单调递增，导致无法识别局部的错误执行（价值应下降）和后续的恢复纠正（价值应回升）。
*   **研究假设**：重试事件提供了自然的价值局部结构——执行偏离正常路径时价值下降，而纠正行为的开始标志着向可行状态的恢复。

### 3. 方法设计详解
*   **流程总结**：
    1.  **全局进度监督**：对常规路径进行绝对进度回归，保证全局价值的基准趋势。
    2.  **重试关键点定位**：仅标注纠正行为的开始瞬间，无需全段标注。
    3.  **偏好对构造**：在重试点周围划分“前(Pre)”、“近(Near)”、“后(Post)”三个窗口，构造`(h+, h-)`偏好对，强迫模型学习“正常 > 错误”和“恢复 > 错误”的序关系。
    4.  **软窗口加权**：根据时间距离对偏好对进行指数加权，减少窗口边界模糊带来的噪声干扰。
    5.  **加权行为克隆**：根据价值改进量（$r_t = V(h_{t+\Delta a}) - V(h_t)$）对动作进行重加权，过滤掉低价值的误操作片段。
*   **模型结构**：基于 VLM 主干网，添加一个离散价值头，通过分类分布预测进度，避免简单的标量回归带来的不稳定性。

### 4. 方法对比分析
*   **本质区别**：从传统的“时间轴单调回归”转向“基于局部偏好序的形状校准”。
*   **创新贡献**：引入重试关键点作为稀疏监督源，无需高昂的密集标注成本，实现了对“错误-纠正”动态过程的量化。
*   **适用场景**：复杂长程操作任务、存在大量人为纠正的混合质量演示数据集。

### 5. 实验分析
*   **关键结论**：在四个真实机器人任务中，ReTVL 显著优于 Robometer 和 RECAP-Value，尤其是在识别错误片段的能力（Drop AUC 提升明显）及下游策略成功率上，平均成功率从 41.25% (Standard BC) 提升至 80%。
*   **优势**：极高的局部错误敏感性；无需全量重标数据，标注成本极低（<1min/traj）。
*   **局限**：依赖于对重试点的标注；假设错误后的恢复是单向的，对不可逆的致命错误处理能力有限。

### 6. 实用指南
*   **实现细节**：
    *   **关键超参**：$\tau_w$ (软窗口衰减率) 和 $T_{pref}$ (偏好比较的温度参数) 是保证局部价值形状的关键。
    *   **数据预处理**：需将 30Hz 原始数据降采样至 5Hz 以减少冗余噪声。
*   **迁移建议**：该方法可直接迁移至任何基于 Value-Guided BC 或 Offline RL 的机器人任务中。只需标注纠正开始时刻，无需改变现有的策略架构。

### 7. 总结
*   **核心思想**：利用重试纠正的瞬间，通过成对偏好学习强行纠正价值函数的非单调形状。
*   **速记版 Pipeline**：
    1.  标注纠正开始的关键点。
    2.  构造纠正前后的价值大小偏好对。
    3.  结合全局进度损失训练价值模型。
    4.  按价值改进量过滤并重加权策略训练数据。

**Key Findings:**

- In this work, we propose ReTVL (ReTry-Supervised Value Learning), a framework for learning mistake-sensitive value functions from mixed-quality robot demonstrations by leveraging retry events as sparse supervision.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24633v1)
- [arXiv](https://arxiv.org/abs/2606.24633v1)

---

<a id='2606.24552v1'></a>
## [Enabling Robust Cloth Manipulation via Inference-Time Simulator-in-the-Loop Refinement](https://arxiv.org/abs/2606.24552v1)

**Authors:** Xin Liu, Yulin Li, Ziming Li, Pengyu Jing, Zhenhao Huang, Bingyang Zhou, Ziqiu Zeng, Siyuan Luo, Chenkun Qi, Fan Shi

**Published:** 2026-06-23

**Categories:** cs.RO

**Abstract:**

Simulator-in-the-loop optimization offers a promising inference-time mechanism for robot manipulation. It uses a physical simulator as a backend rollout engine to evaluate candidate trajectories in parallel and refine nominal actions online, a paradigm proven effective in rigid-body manipulation where state and contact are relatively tractable. We bring this paradigm to real-world cloth manipulation from a single RGB input through three pillars. (i) We design a scalable synthetic-data generation and inference-time rollout pipeline built on FLASH, a deformable-object simulator that provides a practical balance among physical fidelity, numerical stability, and rollout efficiency. (ii) We develop a real-to-sim module, trained purely on synthetic data, that maps a single RGB observation to simulation-compatible cloth state by fusing pretrained visual features with learnable canonical tokens. (iii) We perform online planning by coupling a sparse-mesh rollout backend with prior-guided MPPI, anchored at an offline-distilled policy trajectory, preserving manipulation-relevant deformation and contact while enabling sufficient parallel rollout batches. Real-robot experiments show higher success rates and stronger robustness than baseline methods.

**Analysis:**

作为计算机视觉和机器人学习领域的专家，我为您分析这篇论文《Enabling Robust Cloth Manipulation via Inference-Time Simulator-in-the-Loop Refinement》如下：

### 1. 论文核心贡献总结
该论文提出了一种在推理阶段利用“循环内仿真（Simulator-in-the-loop）”机制来实现鲁棒布料操作的框架。通过将高性能可变形物体仿真器（FLASH）与基于单目RGB输入的实时状态估计相结合，该方法克服了布料非刚性、高自由度以及状态不可观测的难题，实现了比传统策略网络更稳健的在线动态规划。

### 2. 关键创新与方法论
该工作的创新之处在于成功将“模型预测控制（MPC）”的思想引入到复杂的柔性物体操作中，其核心技术栈包括：
*   **高效的仿真后端 (FLASH)：** 在物理保真度、数值稳定性和计算效率之间取得了极佳的平衡，使得大规模并行rollout成为可能。
*   **Real-to-Sim 状态重构：** 摆脱了对深度相机的强依赖，通过预训练视觉特征与可学习的“规范化token（Canonical tokens）”融合，将单张RGB图映射为仿真器可用的布料状态。
*   **先验引导的MPPI（模型预测路径积分控制）：** 利用离线训练的策略作为先验锚点（anchor），在推理阶段利用仿真器在线修正动作，解决了纯学习方法在大规模布料变形下的泛化难题。

### 3. 对计算机视觉领域的潜在影响
该研究的核心价值在于**将“感知”与“物理推理”紧密耦合**。对于计算机视觉领域，它展示了：
*   **从单纯的视觉预测转向物理一致性推理：** 证明了即便输入是简单的RGB图像，只要具备良好的状态估计模块和仿真反馈，也能解决极具挑战性的柔性物体操作任务。
*   **显式仿真反馈的价值：** 在“端到端黑盒模型”大行其道的背景下，该论文重申了将物理仿真作为先验知识嵌入推理流程的重要性，这为解决机器人任务中的“分布外（OOD）”泛化问题提供了一种范式。

### 4. 受益的相关领域与应用
*   **家庭服务机器人：** 如自动叠衣服、整理床铺等家务机器人。
*   **医疗手术机器人：** 对软组织（如皮肤、器官）的精密操纵。
*   **复杂场景下的自动装配：** 如汽车制造中的软质内饰安装、包装业中的布袋处理等。
*   **数字孪生与仿真加速：** 高效的实时物理引擎和感知融合算法可广泛应用于数字孪生系统的闭环控制中。

### 5. 可推断的局限性
*   **视觉遮挡问题：** 尽管使用了tokens处理，但单目RGB输入在极度复杂的折叠或严重自遮挡情况下，状态重构的精度可能受限。
*   **仿真与现实的Gap（Sim-to-Real Gap）：** 虽然使用了高效仿真器，但布料的物理特性（摩擦力、材质参数、空气阻力）极其复杂，在动态过程中的高精度物理匹配仍是巨大挑战。
*   **在线计算瓶颈：** 尽管优化了效率，但“循环内仿真”本质上需要大量的并行Rollout，这要求推理端具备较强的GPU算力，对于低成本、低功耗边缘侧设备的部署可能存在困难。

**总结建议：**
这篇论文的有趣之处在于它**拒绝了简单的端到端映射路径**，转而通过构建一个“感知-推理-优化”的闭环系统来解决不确定性。这代表了当前机器人学习从“模仿学习”向“基于物理认知的决策”进化的重要方向。对于从事视觉感知、位姿估计以及柔性物体动力学的研究人员来说，该工作在状态表征和在线规划的结合上具有极高的参考价值。

**Key Findings:**

- (ii) We develop a real-to-sim module, trained purely on synthetic data, that maps a single RGB observation to simulation-compatible cloth state by fusing pretrained visual features with learnable canonical tokens.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24552v1)
- [arXiv](https://arxiv.org/abs/2606.24552v1)

---

<a id='2606.24539v1'></a>
## [PointVG-R: Internalizing Geometric Reasoning in MLLMs for Precise Pointing Localization via Visual Chain of Thought](https://arxiv.org/abs/2606.24539v1)

**Authors:** Ling Li, Bowen Liu, Zinuo Zhan, Jianhui Zhong, Ziyu Zhu, Bingcai Wei, Kenglun Chang, Zhidong Deng

**Published:** 2026-06-23

**Categories:** cs.CV

**Abstract:**

Pointing-based visual grounding requires models to precisely locate target objects by deciphering complex spatial relationships between the visual scene and pointing gestures. Traditional methods typically encode input images into static feature representations and perform reasoning primarily within the linguistic domain, often overlooking the rich perceptual cues and explicit spatial geometry inherent in images. In this study, we aim to mitigate the cognitive vulnerability of models in interpreting gestural spatial relations by proposing PointVG-R, a reasoning-guided Multi-modal Large Language Model (MLLM). PointVG-R introduces geometric-aware reasoning for pointing-based grounding, enabling the model to think with images through the strategic integration of Reinforcement Learning (RL) and cold-start data. Specifically, we design a novel geometric reasoning pipeline that simulates the iterative cognitive process humans employ when interpreting pointing gestures. Furthermore, we construct EgoPoint-CoT, a high-quality visual Chain-of-Thought (CoT) dataset featuring detailed reasoning trajectories to guide the model via Supervised Fine-Tuning (SFT) and RL. To address the varying quality of learning signals encountered during training, we further propose an Adaptive Importance Weighting strategy based on Group Variance, which dynamically adjusts reward signals to optimize the learning process. Experimental results demonstrate that PointVG-R achieves SOTA performance, outperforming the baseline by $\textbf{15.86}$ points in mIoU. Extensive ablation studies further validate the efficacy of our proposed modules. Code: https://github.com/lingli1724/PointVG-R.

**Analysis:**

### 1. 摘要翻译
指向性视觉定位需要模型通过解析视觉场景与指向手势之间的复杂空间关系来精确定位目标物体。传统方法通常将输入图像编码为静态特征，主要在语言域内进行推理，忽视了图像中蕴含的丰富感知线索和显式空间几何。针对传统多模态大模型（MLLM）在处理复杂空间关系时存在的认知脆弱性，我们提出了PointVG-R，一种通过视觉思维链（Visual Chain-of-Thought, V-CoT）内化几何推理以实现精确指向定位的推理引导型MLLM。PointVG-R不仅输出最终边界框，还展示了从手部定位和几何射线估计到目标语义识别的完整逻辑轨迹。这种“以图思考”的范式使其能够克服传统模型的黑盒限制，实现更稳健的空间理解。

### 2. 方法动机分析
- **驱动力**：解决第一人称视角下，现有MLLM仅将指向手势视为“视觉提示”而导致的定位不准、幻觉偏差和逻辑推理缺失问题。
- **痛点**：传统MLLM缺乏对显式几何约束的理解，容易受显著性偏差干扰，且无法处理复杂交互中的动态空间关系。
- **研究假设**：将指向定位建模为一个显式的、分步的“视觉思维链”推理过程（手部检测→关键点提取→射线估计→目标对齐），能显著增强模型对几何逻辑的建模能力。

### 3. 方法设计详解
**方法Pipeline**：
1. **手部检测**：识别交互源，定位手部框（$r_1$）。
2. **关键点提取**：预测手根部与指尖关键点（$r_2$），明确指向方向。
3. **几何射线构建**：利用工具调用（Tool Call）功能绘制射线，提取射线对齐的特征区域，并与全局特征融合（$r_3$）。
4. **目标对齐**：在几何线索与全局上下文约束下，预测目标物体边界框（$r_4$）。

**模型结构**：基于Qwen2.5-VL-7B，通过两阶段训练（CoT-SFT预热 + 自适应GRPO对齐）实现。
**关键算法 - 自适应奖励策略**：
- **核心逻辑**：针对RL中样本质量不均匀问题，引入**组方差（Group Variance）**作为衡量 rollout 质量的指标。
- **计算公式**：计算同组内样本奖励的方差，利用该方差动态赋予权重（$w_i \propto \sqrt{\text{Var}(\mathcal{R})}$）。高方差组反映了样本间的辨别度，因此被赋予更高权重，从而引导模型聚焦高质量逻辑轨迹。

### 4. 方法对比分析
- **本质区别**：从传统的端到端（End-to-End）直接预测映射，转向了结构化的、显式的几何逻辑推理轨迹构建。
- **创新贡献**：
    1. **V-CoT范式**：引入结构化推理步骤，使定位过程可追溯、可解释。
    2. **自适应GRPO**：通过组方差加权机制，解决了RL训练中奖励信号噪声大的痛点，提升训练稳定性。
- **适用场景**： egocentric（第一人称）视角下的视觉基础定位、人机交互场景。

### 5. 实验分析
- **关键结果**：在EgoPoint-CoT基准上，PointVG-R较基准模型mIoU提升了15.86个点，显著超越传统SFT方案。
- **主要优势**：几何推理显式化、对小目标和复杂 clutter 环境的鲁棒性强。
- **主要局限**：推理过程增加了解码 token 长度，导致推理延迟增加，对实时性要求极高的AR设备具有挑战。

### 6. 实用指南
- **开源情况**：已开源，代码见 https://github.com/lingli1724/PointVG-R。
- **关键细节**：
    - 训练分两阶段，第一阶段利用LoRA进行高效微调，注入结构化推理偏置。
    - RL阶段使用了Group Relative Policy Optimization (GRPO)，注意在计算重要性权重时需要对组内奖励方差进行归一化。
- **迁移可能性**：该思路（V-CoT + RL优化）可直接迁移至涉及复杂推理的文档分析（如几何证明）或机器人具身导航任务中。

### 7. 总结
- **核心思想**：内化几何推理，通过思维链轨迹增强视觉基础定位。
- **速记版Pipeline**：
    1. 定位交互手部；
    2. 提取指尖指向关键点；
    3. 绘制指向射线以聚焦目标；
    4. 对齐并输出目标边界框。

**Key Findings:**

- Specifically, we design a novel geometric reasoning pipeline that simulates the iterative cognitive process humans employ when interpreting pointing gestures.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.24539v1)
- [arXiv](https://arxiv.org/abs/2606.24539v1)

---

