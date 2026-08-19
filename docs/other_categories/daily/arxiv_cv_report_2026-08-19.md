time: 20260819

# Arxiv Computer Vision Papers - 2026-08-19

## Executive Summary

# ArXiv 计算机视觉日报执行摘要（2026-08-18）

## 一、总体概览与主要趋势

本期 10 篇论文集中体现出计算机视觉研究从“单纯提升精度”向**高效推理、真实世界部署、数据与能力协同设计**转变的趋势，主要可归纳为以下四条主线：

1. **生成模型的效率优化与能力扩展**
   - *From Corpora to Co-Evolving Capabilities* 从数据设计角度讨论通用图像生成能力如何与训练语料共同演化，关注数据—能力之间的协同关系。
   - *AViTS* 通过自适应时空 token 选择，降低动态分辨率生成的计算开销。
   - *LinCa* 使用可学习的分解式特征缓存加速扩散模型。
   - *GenRec* 探索何时应进行重建、何时应进行生成，体现生成模型与显式视觉重建的混合范式。

2. **面向边缘设备和实时系统的视觉计算**
   - *Jetson-ORB-SLAM3* 将 ORB-SLAM3 针对 NVIDIA Jetson 平台进行 GPU 化实现，目标是在边缘设备上保持精度的同时提升效率。
   - *ETHEREAL* 面向高分辨率事件视觉，设计低延迟、事件驱动的图神经网络处理器。
   - *LinCa*、*AViTS* 也属于通过减少冗余计算来提高生成系统吞吐率和可部署性的工作。

3. **自动驾驶与机器人系统的真实世界可靠性**
   - *Plug-and-Play Traffic Element Awareness* 强调在端到端自动驾驶中显式引入交通元素感知模块。
   - *Stability Control for Real World Testing in Autonomous Racing* 关注自动驾驶赛车的现实测试稳定性，体现从模型性能到闭环系统安全和可控性的转向。
   - *PRISM* 建立带有精细接触信息和多模态传感的工业技能数据集，为机器人操作与技能学习提供真实数据基础。

4. **多模态三维理解与高效信息检索**
   - *Memory Tree Guided Key Frame Querying* 通过记忆树引导关键帧查询，面向高效三维问答。
   - 该方向的核心问题是：如何在长视频、三维场景或多帧观测中只检索最有价值的信息，而非对全部视觉输入进行密集处理。

---

## 二、特别值得关注的论文

### 1. *From Corpora to Co-Evolving Capabilities*
这篇论文的重要性在于其关注点位于生成模型的上游：**如何设计能够推动模型能力持续发展的数据体系**。相比单纯扩大数据规模，能力中心化的数据设计可能更有助于解决数据分布、能力覆盖和训练效率之间的矛盾。建议重点关注其能力划分方式、数据构建流程以及如何验证数据与能力的协同演化。

### 2. *ETHEREAL*
该工作将算法、图神经网络和专用硬件结合起来，针对事件相机的异步数据特性进行端到端设计。其“低延迟 + 高分辨率 + 边缘部署”定位尤其具有工程价值，可能对机器人、无人机和高速视觉系统产生影响。建议重点阅读其硬件架构、事件流处理机制、能耗与延迟评测。

### 3. *AViTS*
动态分辨率生成是当前视觉生成系统降低成本的重要方向。自适应时空 token 选择直接针对 token 数量和时空冗余问题，具有较强的通用性。该方法是否能在保持生成质量的同时稳定减少计算量，以及 token 选择是否会影响细节和时序一致性，是值得重点考察的内容。

### 4. *GenRec*
“在哪里重建、在哪里生成”的思想具有较强的概念吸引力。它可能代表一种更灵活的混合式视觉推理策略：对结构明确的区域采用重建，对不确定或缺失区域使用生成。该思路对于三维重建、视图合成和场景补全具有潜在意义，建议关注其决策机制以及生成误差是否会引入幻觉。

### 5. *PRISM*
高质量、接触丰富的真实工业技能数据仍然相对稀缺。该数据集如果确实覆盖精细接触、多模态传感和真实工业操作，将对机器人模仿学习、操作策略学习和多模态世界模型具有较高参考价值。建议重点查看数据规模、传感器配置、任务多样性和许可协议。

---

## 三、正在浮现的研究方向与技术

- **能力驱动的数据工程**：数据集不再只是规模化收集，而是围绕模型能力缺口进行组织、诊断和动态更新。
- **Token/特征级稀疏化**：自适应 token 选择、特征缓存和关键帧查询成为降低视觉模型推理成本的共同技术路线。
- **生成与重建的协同**：未来系统可能根据观测确定性动态切换显式重建与生成式预测，而非固定采用单一范式。
- **算法—硬件协同设计**：事件视觉、图神经网络和边缘处理器的联合优化，显示视觉模型部署正在从软件优化走向系统级设计。
- **端到端系统中的显式结构先验**：交通元素、稳定性控制和场景记忆等模块说明，纯粹端到端学习可能会与可解释的中间表示、控制约束和检索机制结合。
- **真实世界数据与闭环评测**：工业操作、自动驾驶赛车和边缘 SLAM 等工作共同强调现实环境中的延迟、鲁棒性、安全性和传感器噪声，而不仅是离线基准成绩。

---

## 四、建议优先阅读全文的论文

### 第一优先级：关注研究趋势与方法创新
1. **From Corpora to Co-Evolving Capabilities**  
   适合关注生成模型、数据策展、基础模型训练策略的研究人员。
2. **AViTS**  
   适合研究视频生成、扩散模型、动态计算和高效 Transformer 的读者。
3. **GenRec**  
   适合关注三维视觉、场景重建、生成式建模和视觉推理的读者。
4. **ETHEREAL**  
   适合边缘 AI、事件视觉、神经形态计算和硬件加速方向的研究人员。

### 第二优先级：关注真实系统与工程落地
5. **PRISM**  
   适合机器人学习、工业视觉和多模态数据集研究。
6. **Jetson-ORB-SLAM3**  
   适合 SLAM、机器人导航和嵌入式视觉部署。
7. **Stability Control for Real World Testing in Autonomous Racing**  
   适合自动驾驶控制、安全验证和 sim-to-real 研究。
8. **Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving**  
   适合端到端自动驾驶及结构化感知融合研究。

### 第三优先级：关注效率优化的具体技术
9. **LinCa**  
   适合扩散模型加速、缓存机制和推理优化研究。
10. **Memory Tree Guided Key Frame Querying**  
    适合三维问答、长序列视觉检索和记忆增强模型研究。

## 一句话总结

本期论文的共同方向是：**让视觉模型更高效、更具结构意识，并真正适用于边缘设备、自动驾驶、机器人和工业环境；与此同时，数据设计、检索机制以及生成—重建协同正逐渐成为提升通用视觉能力的关键组成部分。**

---

## Table of Contents

1. [From Corpora to Co-Evolving Capabilities: Capability-Centric Data Design for Generalist Image Generation](#2608.18076v1)
2. [Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving](#2608.18035v1)
3. [Memory Tree Guided Key Frame Querying for Efficient 3D Question Answering](#2608.18009v1)
4. [AViTS: Adaptive Spatiotemporal Token Selection for Efficient Dynamic-Resolution Generation](#2608.17995v1)
5. [LinCa: Accelerating Diffusion Models via Learnable Decomposed Feature Caching](#2608.17973v1)
6. [PRISM: Precision and contact-rich Real-world Industrial Skill dataset with Multimodal sensing](#2608.17962v1)
7. [Jetson-ORB-SLAM3: Accuracy-Preserving GPU Implementation for Edge Computing Devices](#2608.17874v1)
8. [GenRec: Knowing Where to Reconstruct and Where to Generate](#2608.17832v1)
9. [ETHEREAL: A 25.6-$μ$s/inf. Low-latency Event-driven Graph-neural-network Processor for High-resolution Vision at the Edge](#2608.17787v1)
10. [Stability Control for Real World Testing in Autonomous Racing](#2608.17779v1)

---

## Papers

<a id='2608.18076v1'></a>
## [From Corpora to Co-Evolving Capabilities: Capability-Centric Data Design for Generalist Image Generation](https://arxiv.org/abs/2608.18076v1)

**Authors:** Xingjian Wang, Zhao Wang, Taihang Hu, Jun Zheng, Qing Jin, Qinye Zhou, Zhengtao Wu, Yongchao Du, Zuan Gao, Chao Lin, Yefeng Shen, Xiaoli Xu, Zhengze Xu, Hao Yan, Yuhang Yu, Mingzhou Zhang, Mengting Chen

**Published:** 2026-08-18

**Categories:** cs.CV, cs.AI

**Abstract:**

Large-scale image generation has benefited from advances in data scale, quality, rebalancing, and recaptioning, yet conventional pipelines typically optimize task-specific datasets in isolation. A central challenge is not only how to curate each task-specific corpus, but also how to organize heterogeneous supervision according to the dependencies among generative capabilities. We present a \textbf{capability-driven data infrastructure} that couples capability-specific supervision construction with capability-aligned curriculum scheduling. Its three specialized yet interoperable data engines build complementary relational supervision for text-image grounding, inter-image transformation, and image-knowledge association, while caption experts align T2I and editing supervision across tasks and granularities. A multi-stage curriculum jointly evolves task composition, visual-concept distribution, data quality, and image resolution along the dependency order of capability acquisition, with capability-aware evaluation closing the loop through targeted retrieval, expert construction, and gap-aware resampling. At scale, the framework curates a 440M-image T2I corpus, 120M editing pairs, and over 27M image-entity pairs. With this infrastructure, we train multimodal diffusion models at two scales from scratch, with 3B and 6B sizes respectively. We conduct quantitative evaluation on CPI-Bench, along with qualitative evaluations across diverse text-to-image and editing scenarios. Experimental results present broad visual coverage, versatile rendering, and effective transfer across generative capabilities.

**Analysis:**

## 1. 主要贡献概述

本文提出一种**以生成能力为中心的数据设计与训练基础设施**，不再将文本到图像生成、图像编辑和图像知识理解视为彼此独立的数据工程任务，而是根据能力之间的依赖关系进行联合构建和课程式调度。该框架规模化构建了约 **4.4亿张 T2I 图像、1.2亿组编辑数据和超过2700万组图像—实体关联数据**，并据此从头训练了3B和6B规模的多模态扩散模型，展示了较强的跨任务迁移、视觉覆盖和生成能力。

## 2. 关键创新与方法

### 2.1 从“数据集中心”转向“能力中心”

传统方法通常针对单一任务分别收集和优化数据，例如单独构建文本到图像数据集或图像编辑数据集。本文的核心转变是：先分析模型需要获得哪些生成能力，以及这些能力之间的依赖关系，再反向设计数据和训练流程。

这意味着数据不再仅按来源、任务或规模组织，而是按诸如以下能力进行组织：

- 文本与图像之间的语义对齐和细粒度 grounding；
- 图像到图像的结构、风格和内容变换；
- 图像与实体、概念及外部知识之间的关联；
- 跨任务、跨粒度的视觉概念理解与表达。

### 2.2 三类互补的数据引擎

摘要提到三个专门但可互操作的数据引擎，分别构建不同形式的关系监督：

1. **文本—图像 grounding 监督**  
   用于学习文本描述与图像区域、对象、属性及关系之间的对应，有助于提升组合式生成、实体控制和细粒度语义遵循能力。

2. **图像—图像变换监督**  
   通过编辑对、变换对等数据学习图像内容、结构、风格和属性的可控变化，直接服务于图像编辑和跨模态变换。

3. **图像—知识/实体关联监督**  
   将图像与实体、概念或知识信息连接起来，可能有助于生成模型获得更丰富的世界知识、实体识别和概念表达能力。

这三类监督的价值在于，它们不是简单增加样本数量，而是从不同关系层面补充模型的生成能力。

### 2.3 统一的 caption 专家与跨任务对齐

摘要指出，caption experts 被用于对齐 T2I 和编辑监督，并覆盖不同粒度。这一设计可能包括：

- 对图像进行更准确、细粒度和结构化的描述；
- 为编辑前后图像提供变化区域或变化语义；
- 使文本到图像数据与图像编辑数据共享相对一致的语义空间；
- 将对象级、属性级、关系级和全局场景级描述连接起来。

这对解决生成模型中的一个常见问题很重要：T2I 训练中的文本描述与编辑训练中的指令描述往往风格、粒度和语义结构不一致。

### 2.4 能力依赖驱动的多阶段课程学习

论文并非静态混合所有数据，而是根据能力获得的依赖顺序逐步调整：

- 任务构成；
- 视觉概念分布；
- 数据质量；
- 图像分辨率。

这种课程设计可能先学习基础的视觉—语言对齐和常见概念，再逐步加入更复杂的组合关系、编辑操作、长尾实体以及高分辨率图像。其重要之处在于，数据分布和训练难度是动态变化的，而不是简单地按照固定比例采样。

### 2.5 能力感知的闭环评估与重采样

框架通过针对性检索、专家构造数据和差距感知重采样形成反馈闭环：

1. 评估模型在某项能力上的不足；
2. 定位对应的概念、任务或数据缺口；
3. 构造或检索更适合的训练样本；
4. 重新采样并继续训练。

这比单纯依赖总体 FID、CLIP 分数或固定验证集更接近“能力工程”，因为它试图直接优化模型的薄弱能力。

## 3. 对领域的潜在影响

### 3.1 提供了通用图像生成数据工程的新范式

本文的潜在重要性不只是数据规模大，而是提出了一个更系统的观点：**通用生成模型的性能瓶颈可能来自能力之间缺乏协同的数据组织，而不仅仅是数据数量不足**。这种能力依赖建模可能成为大规模视觉模型数据构建的重要方向。

### 3.2 有助于提升多能力迁移

T2I、图像编辑和图像知识理解之间存在共享的视觉概念和语义关系。联合设计监督可以减少不同任务之间的数据分布割裂，从而改善：

- T2I 模型对编辑指令的迁移；
- 编辑模型对复杂文本描述的理解；
- 实体和概念知识对图像生成的支持；
- 多任务训练中的正迁移。

### 3.3 推动数据质量评估从“样本级”走向“能力级”

传统数据清洗通常关注图像美观度、文本相似度、重复率和安全性。本文更进一步，强调数据是否覆盖某种能力、是否填补模型短板。这有可能推动新的数据指标和评估体系，例如概念覆盖率、组合泛化能力、编辑可控性和能力缺口度量。

### 3.4 对大规模基础模型训练具有实践价值

如果其方法有效，它可能降低人工设计复杂数据配方的成本，并为不同模型规模、不同训练阶段提供可迁移的数据调度策略。尤其是在模型已经拥有较强基础能力后，针对性补充长尾概念、细粒度关系和编辑能力，可能比继续盲目扩大通用图像数量更有效。

## 4. 可能受益的相关领域和应用

### 4.1 文本到图像生成

- 复杂场景和多对象组合生成；
- 更严格的属性、空间关系和动作遵循；
- 长尾实体和专业概念的可视化；
- 高分辨率和多风格内容生成。

### 4.2 图像编辑与可控生成

- 局部对象替换、删除和添加；
- 风格迁移与属性修改；
- 保持身份、结构和背景的一致性编辑；
- 基于自然语言的多步编辑；
- 电商商品图、广告创意和影视概念设计。

### 4.3 多模态理解与视觉知识建模

图像—实体关系数据可能有利于：

- 图像实体识别与链接；
- 视觉问答和视觉检索；
- 图像知识库构建；
- 专业领域图像理解；
- 图像生成与外部知识增强。

### 4.4 机器人和具身智能

更强的文本—图像 grounding、图像变换和实体知识关联能力，可用于：

- 机器人根据语言指令理解目标物体；
- 场景重构和未来状态预测；
- 仿真环境或任务场景生成；
- 机器人操作前后的视觉状态建模。

### 4.5 设计、广告和内容生产

统一的 T2I 与编辑能力基础设施适用于：

- 服装和产品设计；
- 游戏、影视及虚拟世界内容制作；
- 工业设计草图生成；
- 个性化营销素材；
- 教育和科学可视化。

## 5. 从摘要可以推断的局限性

以下限制并不一定代表论文已经被实验确认，而是根据摘要信息可以提出的合理疑问。

### 5.1 数据规模带来的资源和可复现性问题

4.4亿张图像、1.2亿组编辑对以及2700万组图像—实体对意味着极高的数据存储、过滤、标注和训练成本。即使方法有效，许多研究者也难以复现完整基础设施，因此其收益可能部分来自巨大的工程投入和算力，而不完全来自能力中心的数据组织方法。

### 5.2 缺少对单个组件贡献的明确说明

摘要没有说明以下关键消融结果：

- 三类数据引擎分别贡献多少；
- caption experts 是否是主要收益来源；
- 课程调度相对于固定混合比例提升多少；
- 闭环重采样是否优于普通难例挖掘；
- 3B和6B模型是否受益程度一致。

如果缺少充分消融，很难判断性能提升究竟来自数据规模、数据质量、模型规模，还是能力驱动的组织策略。

### 5.3 数据质量和自动标注误差可能成为瓶颈

如此大规模的数据很可能大量依赖自动 caption、视觉语言模型或专家模型生成标注。这些标注可能存在：

- 物体属性和空间关系错误；
- 编辑指令与真实变化不一致；
- 实体链接错误；
- 过度描述或幻觉；
- 数据分布和风格偏差。

如果错误监督被重复放大，能力中心的课程设计反而可能将模型引向错误的概念关联。

### 5.4 “能力”定义和测量仍可能不充分

摘要中提到 capability-aware evaluation，但没有详细说明能力如何形式化、如何分解以及如何避免指标与训练数据高度重合。若评估主要使用与数据构建流程相近的检索器或专家模型，可能存在评估偏置，难以证明模型真正具备独立的组合泛化和世界知识能力。

### 5.5 泛化性和跨分布性能尚不明确

论文主要报告 CPI-Bench 及定性评估，但摘要没有提到：

- 不同语言和文化分布上的性能；
- 专业领域图像上的泛化；
- 长尾和未见组合；
- 真实用户编辑需求；
- 对抗性或歧义指令；
- 与最新开源或闭源模型的公平比较。

因此，模型是否真正成为“通用”生成器，还需要更多跨数据集、跨领域和跨语言验证。

### 5.6 版权、安全和社会偏差问题

大规模图像和实体数据可能涉及：

- 图像版权与许可来源；
- 人脸、个人信息和敏感属性；
- 训练数据中的性别、地域和文化偏见；
- 有害或误导性内容生成；
- 受保护艺术风格和身份模仿。

摘要没有讨论数据治理、隐私保护、版权合规和安全过滤，这些对于实际部署尤其重要。

### 5.7 多任务联合训练可能存在负迁移

T2I、编辑和图像知识关联虽然共享视觉语义，但目标并不完全一致。例如，编辑任务强调保持未修改区域，T2I 任务则允许整体重新生成；知识关联强调事实一致性，而艺术生成可能追求开放式表达。不同监督之间可能产生冲突，需要进一步说明模型如何处理任务权衡和负迁移。

## 总体评价

这项工作的趣味性在于，它把大模型数据构建从“收集更多、更干净的图像”提升为“围绕能力依赖设计监督、课程和反馈闭环”。如果实验能够证明能力驱动的数据组织在控制算力和数据总量后仍然显著优于传统混合策略，那么它将对通用图像生成模型的数据配方、跨任务迁移和能力评估产生较大影响；但其真正的普适性仍取决于详细消融、独立评估、数据治理和跨分布泛化结果。

**Key Findings:**

- We present a \textbf{capability-driven data infrastructure} that couples capability-specific supervision construction with capability-aligned curriculum scheduling.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.18076v1)
- [arXiv](https://arxiv.org/abs/2608.18076v1)

---

<a id='2608.18035v1'></a>
## [Plug-and-Play Traffic Element Awareness for End-to-End Autonomous Driving](https://arxiv.org/abs/2608.18035v1)

**Authors:** Zongzheng Zhang, Jijun Wang, Saining Zhang, Shuo Wang, Yiru Wang, Hai Yang, Yang Chen, Yuwen Heng, Hao Sun, Anqing Jiang, Hao Zhao

**Published:** 2026-08-18

**Categories:** cs.CV

**Abstract:**

Traffic elements such as traffic lights and road signs play a fundamental role in human driving decisions and should naturally influence end-to-end driving performance. However, existing end-to-end driving research predominantly focuses on dynamic road participants (e.g., vehicles and pedestrians), while the role of traffic elements remains largely unexplored. The community still lacks a systematic study quantifying their impact, largely because public datasets rarely provide structured traffic-element annotations and modern driving systems vary widely in architecture and training paradigm. In this work, we present the first systematic investigation of traffic element awareness for end-to-end autonomous driving. We construct a unified research infrastructure by augmenting multiple public driving datasets with comprehensive traffic-element annotations. To support diverse model families, we adopt a minimal and universal integration design that incorporates traffic-element signals into existing pipelines in a plug-and-play manner with negligible architectural modification. We evaluate this design across modern paradigms, including perception-prediction-planning pipelines, vision-language-action models (VLA), regression-based planners, diffusion-based policies, and trajectory-scoring frameworks, on nuScenes, NAVSIM-v1, NAVSIM-v2, and Bench2Drive. Across all paradigms and datasets, this simple integration consistently improves driving performance, demonstrating that traffic element awareness provides a robust and generalizable signal for end-to-end driving systems. Notably, on the challenging NAVSIM-v2 benchmark, our approach significantly improves state-of-the-art architectures and data pipelines, establishing a new state of the art.

**Analysis:**

## 1. 摘要翻译
交通元素（如红绿灯、道路标志）深刻影响人类驾驶决策，但现有端到端自动驾驶研究主要关注车辆、行人等动态参与者，较少系统研究交通元素的作用。本文构建了带交通元素标注的多数据集基础设施，并提出一种轻量、通用的即插即用集成方式：通过辅助3D交通元素监督，以及可选的车道拓扑条件，将交通元素信息接入不同端到端规划器。该方法在nuScenes、NAVSIM-v1/v2和Bench2Drive上覆盖回归、扩散、轨迹评分、感知-规划及VLA等范式，均获得稳定提升，并在NAVSIM-v2上达到新的最佳结果。

## 2. 方法动机分析
**驱动力：**红绿灯和道路标志不是普通视觉目标，而是直接约束“能否通行、能否转弯”的规则信号。仅依赖几何特征或动态目标，模型可能生成拓扑上可行、但违反交通规则的轨迹。  
**现有痛点：**公共数据集缺少结构化交通元素及其3D位置；交通元素空间上稀疏，容易被车辆、道路等密集特征淹没；不同规划器结构差异大，单一方法结论难以推广。  
**核心假设：**如果强制模型显式学习“交通元素的位置、类别及其控制的车道”，规划器就能获得比全局深度或普通语义分割更具决策价值的规则信息。

## 3. 方法设计详解
### Pipeline
1. **构造3D交通元素。**从前视图输入进行2D检测和单目深度估计；取检测框中心像素，通过深度与相机内参反投影到3D，再利用LiDAR点进行几何校正，最后经外参变换得到自车坐标系中的3D中心。NAVSIM缺少标注，则用OpenLane-V2训练的YOLO检测器生成伪标签。  
2. **辅助3D监督。**在原规划器的BEV/视觉特征上增加独立TE分支，同时预测3D位置和类别。位置采用L1损失，类别采用Focal Loss：
\[
L=L_{\rm plan}+L_{\rm aux}+\lambda_{\rm TE}(L_{\rm loc}+L_{\rm cls})
\]
其目的不是替代规划损失，而是迫使共享特征为稀疏、规则关键目标分配专门表征能力。  
3. **交通元素接入规划。**不同模型采用相应形式：VAD/Orion使用TE query或特征；LTF、DiffusionDrive、DrivoR预测BEV TE热图，再经MLP投影并与BEV记忆拼接。作者发现稀疏目标不适合平均池化，采用最大池化保留局部峰值；直接拼接通常优于交叉注意力。  
4. **拓扑条件建模。**预测车道-车道（LCLC）和车道-交通元素（LCTE）邻接关系；从全局图中筛选与自车车道及其后继车道相关的局部子图，再转写为结构化文本，如“当前车道前方有红灯，车道仅连接直行车道”。文本经冻结BERT编码为拓扑token，与视觉/BEV特征拼接后共同生成轨迹。  
5. **联合输出。**规划解码器根据场景特征、TE特征和拓扑特征预测未来自车轨迹；推理时使用模型预测的TE，而非直接读取标注。

### 模型协同逻辑
TE提供局部、可执行的规则线索；拓扑提供“该信号是否控制当前车道”及可行方向。二者分别解决“看见什么”和“它是否与我有关”，从而避免无关红灯导致误刹车。

## 4. 方法对比与创新
本质区别在于：作者不是设计新的规划器，也不是加入固定规则后处理，而是把交通元素作为跨架构的辅助学习信号和规划条件。创新主要包括：统一的3D TE构造流程；适用于多种规划范式的轻量接入；自车相关拓扑筛选；证明交通标志与交通灯同样关键。最佳场景是复杂路口、规则密集区域和需要区分多条车道控制关系的环境。缺点是仍依赖检测和深度质量，且结构化语言编码未必比专用图模型更具可解释性或效率。

## 5. 实验分析
作者在四个数据集、六类规划器上进行开环和闭环验证。代表性结论是：NAVSIM-v2中LTF的EPDMS由25.1提升至28.9；Bench2Drive中VAD的Driving Score由42.3提升至56.4。  
主要优势：跨模型稳定增益、改动小、运行开销低，并显著改善安全与规则遵循。  
主要局限：NAVSIM依赖伪标签；恶劣天气和夜间深度误差会削弱效果；当前主要使用单帧状态，未显式建模信号时序变化。

## 6. 实用指南
论文提供项目主页，但正文未明确确认完整代码已公开。复现重点是：训练高质量2D TE检测器；完成深度-LiDAR融合；独立TE头使用Focal Loss；采用最大池化和特征拼接；拓扑只保留自车相关部分。VAD示例训练20轮、AdamW、学习率 \(2\times10^{-5}\)；LTF中热图用高斯核生成，并使用Focal Loss。该思想可迁移到机器人导航、具身智能和工业视觉决策：将稀疏但高价值的规则对象显式建模，再作为辅助监督或条件输入。

## 7. 总结
**核心思想：**让规划器显式理解交通规则。  

**速记版Pipeline：**
1. 检测红绿灯和道路标志；  
2. 根据深度与标定恢复其3D位置；  
3. 用独立分支学习位置和类别；  
4. 找出它们控制的自车车道及可行方向；  
5. 将这些规则信息送入轨迹规划器。

**Key Findings:**

- In this work, we present the first systematic investigation of traffic element awareness for end-to-end autonomous driving.
- Notably, on the challenging NAVSIM-v2 benchmark, our approach significantly improves state-of-the-art architectures and data pipelines, establishing a new state of the art.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.18035v1)
- [arXiv](https://arxiv.org/abs/2608.18035v1)

---

<a id='2608.18009v1'></a>
## [Memory Tree Guided Key Frame Querying for Efficient 3D Question Answering](https://arxiv.org/abs/2608.18009v1)

**Authors:** Hsiang-Wei Huang, Fu-Chen Chen, Li-Wu Tsao, Cheng-Han Lee, Che-Chun Su, Lu Xia, Ronghui Peng, Jenq-Neng Hwang, Min Sun, Cheng-Hao Kuo

**Published:** 2026-08-18

**Categories:** cs.CV

**Abstract:**

Answering questions accurately and efficiently in embodied scenarios presents significant challenges due to limited computational and memory resources for Vision Language Model (VLM) inference. Existing methods adopt visual search key frame retrieval method to select critical question-related key frames for VLM input. However, visual search methods are inefficient because they require visual search among thousands of video frames for each individual user query. In this work, we propose a memory tree guided key frame selection paradigm for efficient 3D question answering in embodied scenarios. Our method leverages a compact and reusable 3D scene representation, termed MemTree3D, which supports real-time online construction leveraging camera 6-DoF poses. MemTree3D captures multi-level 3D scene information, enabling a Large Language Model to efficiently query and retrieve question-relevant key frames through our scoring-based frame selection without reprocessing the entire video stream. On OpenEQA, our method improves the LLM-Match of GPT-4o by 17.4%, LLaVA-OneVision-7B by 5.8%, outperforms existing visual search methods. Our code is available at https://github.com/hsiangwei0903/MemTree3D

**Analysis:**

## 1. 摘要翻译

在具身场景中，由于视觉语言模型（VLM）推理受到计算量和显存限制，如何准确、高效地回答问题仍然具有挑战性。现有方法通常通过视觉搜索从视频中检索与问题相关的关键帧，但每个查询都需要在数千帧中重新搜索，效率较低。本文提出一种由记忆树引导的关键帧选择范式，用于高效三维问答。方法构建紧凑且可复用的三维场景表示 **MemTree3D**，利用相机六自由度位姿进行实时在线构建。MemTree3D编码多层次场景信息，使大语言模型（LLM）能够通过基于评分的帧选择，高效检索与问题相关的关键帧，而无需重新处理完整视频流。在OpenEQA上，相比均匀采样，本文使GPT-4o的LLM-Match提升17.4%，使LLaVA-OneVision-7B提升5.8%，并优于现有视觉搜索方法。

## 2. 方法动机分析

**驱动力与痛点：** 多帧VLM直接输入长视频，计算和显存随帧数快速增长；场景图虽高效，却只保留粗粒度物体信息，丢失颜色、状态和空间细节；视觉搜索能够保留原始图像，但每次提问都要对全部帧运行检测器，延迟随视频长度增长，且检测失败后缺乏恢复机制。

**核心假设：** 一次性构建“空间组织+时间关联”的轻量场景记忆，LLM即可先推断问题可能对应的场景位置，再从少量候选位置中选择视觉证据；因此，查询阶段不必重新扫描完整视频。

## 3. 方法设计详解

### 3.1 总体流程

输入为带相机6-DoF位姿的3D扫描视频，输出为VLM生成的答案：

1. **在线构建MemTree3D。**  
   使用YOLO-World逐帧检测物体，使用BoT-SORT建立跨帧轨迹。按照相机位姿变化进行空间分段：若当前帧与上一个位置节点基准帧之间的平移变化超过1.5米，或旋转变化超过45°，则创建新的Location Node。相比固定时间窗口，这种方式更符合具身相机的真实空间移动。

2. **组织三级树结构。**  
   - **Location Node：** 一个空间连续区域或时间片段，具有唯一位置ID。  
   - **Object Node：** 该区域内某个物体的跟踪轨迹，例如桌子、镜子、沙发。  
   - **Detection Node：** 具体帧中的检测结果，包含时间戳、边界框和置信度。  
   构建阶段保留完整三级信息，但发送给LLM时只序列化Location Node和Object Node，以控制上下文长度。

3. **LLM生成时空线索。**  
   将问题和MemTree3D的JSON描述输入LLM。LLM选择最可能包含答案的前k个位置，默认k=3；同时将物体分成两类：  
   - **关键物体（Okey）：** 与问题直接相关的对象；  
   - **提示物体（Ocue）：** 与关键物体共现、可帮助判断房间或空间关系的对象。  
   例如“手巾在哪里”可能选择包含马桶、镜子、洗手池的区域，即使“手巾”本身未被检测到。

4. **候选位置内评分选帧。**  
   对每个选中的位置，遍历其中的Detection Node，汇总关键物体和提示物体的检测置信度。关键物体权重显著更高，实验中关键物体与提示物体权重比为10:1。可概括为：
   \[
   S(f)=10\sum_{o\in O_{key}}c_{f,o}
   +\sum_{o\in O_{cue}}c_{f,o}
   \]
   其中 \(c_{f,o}\) 是帧f中物体o的检测置信度。每个位置选择得分最高的一帧，从而同时保证问题相关性与空间视角多样性。

5. **VLM回答问题。**  
   将问题和来自不同Location Node的k个关键帧输入VLM，由VLM完成颜色、状态、细粒度属性及空间关系判断。MemTree3D只负责“找证据”，不替代VLM进行最终视觉识别。

### 3.2 设计关键点

该方法的核心不是构建一个用于直接问答的完整场景图，而是构建一个**面向检索的中间记忆结构**。LLM处理高层语义和空间推理，检测器提供可计算的帧级证据，VLM负责最终细粒度识别，三者职责明确。

## 4. 方法对比与创新

**本质区别：** 传统视觉搜索是“问题到来后扫描视频”；本文是“视频采集时预构建记忆，问题到来后查询记忆”。其检索复杂度不再直接依赖原始视频帧数。

**主要创新：**

1. 用6-DoF运动变化而非视觉语义相似度构建空间位置节点，低成本且适合连续具身扫描；
2. 将LLM的空间常识推理用于补偿检测遗漏，而不是完全依赖目标检测结果；
3. 通过关键物体和提示物体的加权置信度进行帧选择，兼顾直接证据与上下文证据；
4. 采用位置级检索，天然鼓励跨视角选帧，避免CLIP等方法重复选择相似视角。

**适用场景：** 长时3D扫描、室内机器人、多轮用户问答、显存有限且需要低查询延迟的具身系统。对“完全未出现在检测类别中的新物体定位”效果较弱。

## 5. 实验分析

作者在OpenEQA、ScanQA和SQA3D上验证方法，并进行帧数、位置节点构建策略、模型组合、运行时间和感知失败鲁棒性实验。

最具代表性的结论是：在OpenEQA上，使用3帧时，MemTree3D使GPT-4o从49.4提升至66.8，使LLaVA-OneVision-7B从49.2提升至55.0；相比检测器驱动的视觉搜索，关键帧选择至少获得69.2%的速度提升。主要优势是一次构建、多次查询、低延迟，并能通过上下文推理缓解漏检。主要局限是新颖物体未进入Object Node时，LLM只能猜测可能位置，无法保证检索到清晰目标。

## 6. 实用指南

论文已开源代码：<https://github.com/hsiangwei0903/MemTree3D>。复现时需准备带6-DoF位姿的视频、YOLO-World、BoT-SORT及ScanNet-200类别；重点设置平移阈值1.5米、旋转阈值45°、位置数k=3、关键/提示物体权重10:1，并注意处理视频末尾检测缓冲区的最终写入。该方法无需训练MemTree3D或微调LLM/VLM，主要成本在在线检测和跟踪。

迁移到导航、视频问答或仓储问答时，可将Location Node替换为地图区域、航迹片段或时间窗口，将Object Node替换为事件、物体状态或动作轨迹，再保留“LLM产生候选区域—评分选择原始证据—VLM/任务模型决策”的框架。

## 7. 总结

**核心思想：** 用可查询的空间记忆替代逐问扫描视频。

**速记版Pipeline：**

1. 根据相机移动把视频切成连续空间区域；  
2. 在区域中记录物体轨迹和逐帧检测结果；  
3. 让LLM根据问题猜测相关区域及关键、辅助物体；  
4. 按检测置信度从不同区域挑选少量代表帧；  
5. 将这些帧交给VLM完成最终回答。

**Key Findings:**

- In this work, we propose a memory tree guided key frame selection paradigm for efficient 3D question answering in embodied scenarios.
- Our method leverages a compact and reusable 3D scene representation, termed MemTree3D, which supports real-time online construction leveraging camera 6-DoF poses.
- On OpenEQA, our method improves the LLM-Match of GPT-4o by 17.4%, LLaVA-OneVision-7B by 5.8%, outperforms existing visual search methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.18009v1)
- [arXiv](https://arxiv.org/abs/2608.18009v1)

---

<a id='2608.17995v1'></a>
## [AViTS: Adaptive Spatiotemporal Token Selection for Efficient Dynamic-Resolution Generation](https://arxiv.org/abs/2608.17995v1)

**Authors:** Haoran Qin, Zhengan Yan, Shikang Zheng, Xiaobing Tu, Jiacheng Liu, Yuqi Lin, Chang Zou, JinShan Liu, Peiliang Cai, Xiantao Zhang, Jinkui Ren, Linfeng Zhang

**Published:** 2026-08-18

**Categories:** cs.CV

**Abstract:**

Diffusion Transformers (DiTs) achieve high-quality generation but are costly due to iterative sampling. Dynamic-resolution sampling reduces early-stage cost by denoising at low resolution; however, uniformly upsampling all latent tokens at resolution transitions incurs redundant computation and may degrade fine-detail consistency. Existing partial upsampling strategies typically rely on local latent structure cues or single-step statistics, making it difficult to jointly capture token-text semantic relevance and token-wise representation dynamics across diffusion steps. We propose AViTS, an adaptive spatiotemporal token selection framework for dynamic-resolution DiTs. AViTS models spatial importance via latent-text attention and temporal importance via token-level feature variation across diffusion timesteps, and fuses them to enable spatiotemporal importance-aware selective upsampling: it prioritizes resolution refinement for critical tokens while deferring less important ones, thereby reducing redundant high-resolution computation and improving the quality-efficiency trade-off. AViTS achieves up to 6.34x on FLUX and nearly 9x FLOPs reduction on Qwen-Image-Edit and FLUX.1-Kontext-dev, orthogonal to distillation, quantization, and feature caching, and reaching 14.76x with distilled models. Code: https://github.com/QHR69/AViTS

**Analysis:**

## 1. 摘要翻译

扩散Transformer（DiT）能够生成高质量图像，但迭代采样带来巨大计算开销。动态分辨率采样通过早期低分辨率去噪降低成本，但在分辨率切换时统一上采样所有潜变量token，会产生冗余计算并损害细节一致性。现有部分上采样方法多依赖局部结构或单步统计，难以同时刻画token与文本的语义相关性及跨扩散步的表示变化。为此，本文提出AViTS：利用潜变量—文本注意力建模空间重要性，以跨时间步的token特征变化建模时间重要性，并融合二者进行选择性上采样，优先高重要性token、延后低重要性token。该方法在FLUX上最高实现6.34×加速，在Qwen-Image-Edit和FLUX.1-Kontext-dev上实现近9× FLOPs削减，并可与蒸馏、量化和特征缓存结合，最高达到14.76×加速。

## 2. 方法动机分析

**驱动力：**高分辨率DiT的token数多，注意力计算昂贵；动态分辨率虽能降低早期成本，但“所有token同时升采样”仍不够精细。  
**现有痛点：**RALU依赖边缘，Fresco依赖通道方差等低层线索，容易把预算分配给背景、纹理或无关边缘；同时，它们通常只观察单步特征，无法反映token的语义条件和持续变化状态。  
**核心假设：**真正值得优先高分辨率处理的token，应同时满足“与文本语义相关”或“在去噪过程中仍剧烈变化”。

## 3. 方法设计详解

### 整体流程

1. **低分辨率初始化。**在目标分辨率潜空间加入噪声，再按空间因子 \(f=2\) 下采样，使token数由 \(M\) 降为 \(M'=M/f^2\)。先执行 \(N_1\) 步低分辨率去噪，建立整体布局。  
2. **多步信号收集。**在随后 \(N_T\) 个低分辨率步骤中，保存每个token的特征快照 \(z_i^{(t_k)}\)，并提取图像token到文本token的注意力。相比单步打分，多步收集能够估计token的动态稳定性。  
3. **计算空间重要性。**将图像token指向所有文本token的注意力，在多个头、层和收集步骤上聚合，得到  
\[
A_i=\frac1{|\mathcal H|}\sum_h\sum_j\alpha_{i,j}^{(h)}.
\]
对于Qwen等MLLM，直接读取图像—文本注意力子矩阵；对于FLUX，则通过hook获得Q/K，重建联合注意力并截取image-to-text区域。  
4. **计算时间重要性。**对token在 \(N_T\) 个快照上的每个通道计算无偏方差，再对通道求平均：  
\[
V_i=\frac1D\sum_d\mathrm{Var}_k(z_{i,d}^{(t_k)}).
\]
方差越大，表示该token仍在快速演化，越需要提前精细化。  
5. **融合与选择。**将 \(A_i,V_i\) 分别归一化到[0,1]，按  
\[
S_i=\alpha\hat A_i+(1-\alpha)\hat V_i
\]
融合，选取前 \(K=\lfloor\rho M'\rfloor\) 个token。  
6. **混合分辨率去噪。**选中的token先正交上采样，未选中的token保持低分辨率，重新注入与坐标相关的噪声，执行 \(N_2\) 步混合分辨率去噪。最后将剩余token全部上采样、重排空间坐标，再执行 \(N_3\) 步全分辨率细化并解码。

本方法的关键不在于改变DiT主体，而在于增加一个“多模态、跨时间”的预算分配器。

## 4. 方法对比分析

**本质区别：**AViTS不是固定分辨率调度，也不是缓存特征，而是动态决定“哪些空间位置先获得高分辨率计算”。它把语义相关性与表示动态性结合起来，属于token级计算资源分配。  
**创新点：**①提出空间—时间双重要性建模；②使用多步token方差而非单步启发式指标；③适配MLLM注意力和FLUX联合注意力两类架构；④无需训练，并可叠加缓存、量化和蒸馏。  
**适用场景：**文本生成、指令编辑、高分辨率生成，尤其适合编辑区域稀疏、背景需保持不变的任务。

## 5. 实验分析

作者在FLUX.1-dev、Qwen-Image-Edit和FLUX.1-Kontext-dev上，用DrawBench、GEdit及延迟/FLOPs评估。代表性结果是：FLUX上约5.45×加速仍保持甚至提升质量；结合步蒸馏最高14.76×。消融表明，注意力或时间方差单独有效，二者融合取得最佳质量—速度折中。  
**优势：**训练免费、语义对齐强、组合性好。  
**局限：**提取注意力和保存多步特征有额外开销；top-K选择是硬切分，可能产生空间边界不连续；极高分辨率、密集细节和文字渲染仍可能失败，且效果依赖注意力可靠性和超参数。

## 6. 实用指南

论文提供代码：<https://github.com/QHR69/AViTS>。复现重点是实现低分辨率三阶段调度、注册注意力hook、缓存 \(N_T\) 个快照、正确执行坐标噪声重注入和token重排。重点调节融合权重 \(\alpha\) 与上采样比例 \(\rho\)：中间值通常优于单一线索；增大 \(\rho\) 可提升质量但降低加速。无需额外训练。迁移到视频时，可将时间方差扩展为帧间与扩散步联合变化；迁移到其他DiT则需获得图像—条件注意力或设计等价的条件相关性指标。

## 7. 总结

**核心思想：**按语义和变化优先精修token。

**速记版Pipeline：**
1. 先用低分辨率建立整体结构；  
2. 连续观察各位置与文本的关联及其变化；  
3. 融合两类分数，挑出最关键区域；  
4. 关键区域先升清晰度，其余区域延后；  
5. 最后统一细化并生成图像。

**Key Findings:**

- We propose AViTS, an adaptive spatiotemporal token selection framework for dynamic-resolution DiTs. AViTS models spatial importance via latent-text attention and temporal importance via token-level feature variation across diffusion timesteps, and fuses them to enable spatiotemporal importance-aware selective upsampling: it prioritizes resolution refinement for critical tokens while deferring less important ones, thereby reducing redundant high-resolution computation and improving the quality-efficiency trade-off.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.17995v1)
- [arXiv](https://arxiv.org/abs/2608.17995v1)

---

<a id='2608.17973v1'></a>
## [LinCa: Accelerating Diffusion Models via Learnable Decomposed Feature Caching](https://arxiv.org/abs/2608.17973v1)

**Authors:** Jinshan Liu, Haoran Qin, Xiaobing Tu, Jiacheng Liu, Jiahui Hu, Zhengan Yan, Yukun Xie, Kerui Shen, Jinkui Ren, Yuqi Lin, Xiantao Zhang, Linfeng Zhang

**Published:** 2026-08-18

**Categories:** cs.CV

**Abstract:**

Diffusion models have achieved remarkable success in image and video generation, yet the high computational cost of iterative sampling remains a critical bottleneck for practical deployment. Feature caching has emerged as a promising acceleration paradigm by reusing or predicting intermediate features across timesteps. However, existing training-free methods apply uniform prediction strategies that cannot adapt to the heterogeneous feature dynamics, causing significant quality degradation under high acceleration ratios. We propose LinCa, a feature caching framework based on learnable invertible networks. LinCa decomposes cached features into sub-components with distinct continuity properties via a lightweight invertible network and applies differentiated prediction orders matched to each component. The strict invertibility guarantees lossless reconstruction back to the original feature space, forming a unified Decompose-Predict-Reconstruct pipeline. By training separate predictors for different models and timestep segments, LinCa adapts to heterogeneous feature dynamics. Experiments on FLUX, Qwen-Image, and HunyuanVideo demonstrate that LinCa, with less than 0.2% additional parameters, significantly outperforms existing methods and maintains near-lossless quality at 5-7x speedup. Code: https://github.com/QHR69/LinCa

**Analysis:**

## 1. 摘要翻译

扩散模型在图像和视频生成中取得了显著成功，但迭代采样的高计算成本仍是实际部署的主要瓶颈。特征缓存通过复用或预测不同时间步的中间特征，成为一种有效的加速范式。然而，现有免训练方法通常采用统一的预测策略，无法适应特征动态的异质性，在高加速比下容易造成明显的质量下降。本文提出 LinCa，一种基于可学习可逆网络的特征缓存框架。LinCa 利用轻量级可逆网络将缓存特征分解为具有不同连续性的子成分，并为各成分匹配不同阶数的预测器。严格的可逆性保证特征能够无损重建，形成“分解—预测—重建”的统一流程。针对不同模型和去噪时间段，作者分别训练预测器，使 LinCa 能适应异质特征动态。在 FLUX、Qwen-Image 和 HunyuanVideo 上，LinCa 仅增加不到 0.2%的参数，即可在 5–7 倍加速下维持近乎无损的生成质量。

## 2. 方法动机分析

**驱动力与痛点：**传统缓存方法每隔 \(N\) 步执行一次完整网络，其余步骤统一采用复用或高阶外推。但作者观察到三种不匹配：不同模型的特征轨迹不同；同一模型不同去噪阶段的连续性不同；即使同一阶段，不同特征维度也可能分别表现为平滑变化或突发跳变。因此，对全部维度使用同一个预测阶数会使稳定维度未被充分利用，或使不稳定维度发生严重外推误差。

**核心假设：**原始特征维度虽然在空间中交错，但经过合适的可学习变换后，可以被重新组织为若干连续性不同的子空间；不同子空间应采用不同预测阶数。

## 3. 方法设计详解

### Pipeline

1. **缓存特征。**在计算步执行扩散模型，缓存末层累积残差特征 \(x_t\in\mathbb R^{N\times D}\)。每隔 \(N\) 步计算一次，其余 \(N-1\) 步跳过主干网络。早期阶段使用 `first_enhance` 预热，避免初始特征不稳定。

2. **可学习分解。**将缓存特征输入分段专属的可逆映射：
\[
z_t=E_\theta^{(s)}(x_t)=[z_t^{(0)},z_t^{(1)},z_t^{(2)}].
\]
网络学习把不同动态特征重新聚集到不同子空间，而不是简单按原始通道切分。

3. **差异化预测。**第0组对应突变、低连续性特征，直接复用最近缓存值：
\[
\hat z^{(0)}_{t-k}=z^{(0)}_t.
\]
第1组使用一阶 Hermite 外推，第2组使用二阶 Hermite 外推。预测依赖最近三个计算步的离散差分，能够利用历史变化趋势和局部曲率。

4. **可逆重建。**将各子空间预测结果拼接后，经 \(E_\theta^{-1}\) 映射回原始特征空间：
\[
\hat x_{t-k}=E_\theta^{-1}([\hat z^{(0)},\hat z^{(1)},\hat z^{(2)}]).
\]
可逆映射本身不丢失信息，误差主要来自子空间预测。

### 模型结构与训练

\(E_\theta\) 由多个可逆块组成，每块包含可逆 \(1\times1\) 卷积和加性耦合层：
\[
v_1=u_1+F(u_2),\quad v_2=u_2+G(v_1).
\]
其逆过程通过减法恢复输入，因此可严格重建。不同模型、不同去噪阶段分别训练参数独立但结构相同的预测器。损失为原空间预测损失与子空间预测损失之和：
\[
L=L_{\rm feat}+\lambda L_{\rm comp}.
\]
前者保证最终特征准确，后者促使网络真正形成适合不同阶数预测的子空间。

## 4. 方法对比分析

LinCa 的本质区别不是提出新的单一外推公式，而是**先学习特征坐标系，再进行按子空间匹配的预测**。复用法相当于所有维度使用0阶预测；TaylorSeer 等方法则对完整特征统一使用高阶预测。LinCa 通过可逆分解解决“维度动态混杂”问题，并通过分时间段训练解决跨阶段差异。它适合多步、高质量扩散图像、视频及编辑模型，尤其适用于较大缓存间隔；极少步模型因冗余有限，收益会下降。

## 5. 实验分析

作者在 FLUX、Qwen-Image、Qwen-Image-Edit、HunyuanVideo，以及蒸馏和 INT8 模型上进行评估。代表性结果是：FLUX.1-dev 达到约 **5.51倍**加速，Qwen-Image 达到 **6.95倍**加速，质量显著优于统一预测基线；消融实验表明，可逆可学习分解和“0/1/2阶混合预测”均是性能关键。

主要优势是质量—速度折中好、预测开销极低、兼容蒸馏和量化。局限在于需要针对每个模型训练额外预测器，依赖预生成特征轨迹；可逆性只保证映射无损，不能保证预测本身无误；网络、分段数和缓存间隔仍需调参。

## 6. 实用指南

论文提供代码：`github.com/QHR69/LinCa`。复现时需先生成约100–200条特征轨迹，再脱离扩散模型权重训练预测器，约1小时、12GB显存即可完成。默认建议：子空间数 \(M=3\)、时间段数 \(S=3\)、损失权重 \(\lambda=1\)、可逆块数 \(L=2\)、隐藏维度128；高阶预测优先使用 Hermite。迁移到其他扩散模型时，只需提取稳定的缓存层特征，按目标模型和时间段重新生成轨迹并训练对应预测器。

## 7. 总结

**核心思想：学习分解特征，再匹配预测阶数。**

**速记版 Pipeline：**
1. 每隔若干步运行模型并保存中间特征；  
2. 用可逆网络把混杂特征重新分成三类；  
3. 不稳定部分直接复用，平滑部分分别做一阶、二阶预测；  
4. 用逆网络恢复原始特征并跳过模型计算；  
5. 按模型和去噪阶段分别训练预测器。

**Key Findings:**

- We propose LinCa, a feature caching framework based on learnable invertible networks.
- Experiments on FLUX, Qwen-Image, and HunyuanVideo demonstrate that LinCa, with less than 0.2% additional parameters, significantly outperforms existing methods and maintains near-lossless quality at 5-7x speedup.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.17973v1)
- [arXiv](https://arxiv.org/abs/2608.17973v1)

---

<a id='2608.17962v1'></a>
## [PRISM: Precision and contact-rich Real-world Industrial Skill dataset with Multimodal sensing](https://arxiv.org/abs/2608.17962v1)

**Authors:** Tengbo Yu, Jiahao Wu, Hanning Wang, Rui Chen, Chuanhou Liu, Chuang Sun, Hangxin Liu

**Published:** 2026-08-18

**Categories:** cs.RO

**Abstract:**

Recent progress in robotic learning has been fueled by large-scale datasets collected in everyday environments. However, most existing datasets emphasize short-horizon, low-contact tasks such as pick-and-place, and therefore do not capture the precision control, force/torque or tactile regulation, and multimodal feedback required for industrial assembly. To address this gap, we introduce PRISM, a large-scale multimodal dataset for contact-rich industrial operations. The dataset spans more than 25 manipulation tasks (e.g., electronic components plug/unplug, conveyor-based sorting) and covers diverse mechanical constraints. PRISM includes more than 5,000 trajectories totaling 45 hours of teleoperated demonstrations, recorded using synchronized multi-view RGB-D, force/torque, tactile, and robot-state measurements. In contrast to datasets collected in household or laboratory settings, PRISM provides a realistic benchmark for multimodal perception and control under high-precision industrial constraints, and serves as a foundation for contact-rich, generalizable manipulation in real-world manufacturing environments. The dataset is open-sourced at: https://tengbo-yu.github.io/PRISM/

**Analysis:**

# 1. 摘要翻译

近年来，机器人学习的发展受益于在日常环境中采集的大规模数据集。然而，现有数据集大多关注短时域、低接触任务，如抓取与放置，无法充分刻画工业装配所需的精密控制、力/力矩或触觉调节，以及多模态反馈。为此，本文提出 PRISM：一个面向接触丰富型工业操作的大规模多模态数据集。数据集涵盖超过25类操作任务，包括电子元件插拔、基于传送带的分拣等，覆盖多种机械约束；共包含5000余条轨迹、总计45小时以上的遥操作示范，并同步记录多视角RGB-D、力/力矩、触觉和机器人状态信息。PRISM为高精度工业约束下的多模态感知与控制提供了真实基准，并为接触丰富、可泛化的现实制造操作研究奠定基础。

# 2. 方法动机分析

**驱动力与痛点：**家庭机器人数据集虽然规模大，却主要包含抓取、搬运、堆叠等短时域任务，缺少插入、装配、卡合等持续接触行为。工业任务中的微小偏差、摩擦、卡滞和过大接触力往往在视觉上难以区分，却能通过力/力矩或触觉信号直接体现。现有高精度数据集又通常受限于单一机器人、单一遥操作方式或较小规模。

**核心假设：**如果在大规模数据中同时引入工业约束、多机器人形态、多种遥操作方式和同步接触传感器，模型就能学习跨任务共享的接触模式与纠错先验，从而提升工业操作的泛化性。

# 3. 方法设计详解

PRISM的“方法”本质上不是新网络，而是一套**面向接触丰富工业操作的数据构建与验证流程**：

1. **任务与平台设计：**任务覆盖NIST装配任务板、包装、汽车部件装配、电子插拔、动态分拣等25类以上操作。使用Franka、Realman和LEJU三类机器人，以及平行夹爪、视触觉夹爪和灵巧手等末端执行器，制造不同运动学和接触模式。
2. **多源遥操作采集：**采用外骨骼、Tracker和VR三种接口。外骨骼记录人—机器人对应的关节运动；Tracker平台支持Franka双臂和视触觉末端；VR平台提供沉浸式第一视角控制。8名志愿者经过训练后采集、筛选、标注并评分。
3. **同步传感：**每条轨迹记录2–4路RGB-D图像、关节角/关节力矩、末端位姿、夹爪状态和六维末端力/力矩；部分轨迹增加30 Hz视触觉图像。RGB-D为15 Hz，机器人状态为15 Hz，六维力/力矩为100 Hz。各模态保留原始时间戳，而非简单强制重采样。
4. **标定与坐标统一：**建立机器人、相机、力传感器和触觉传感器之间的变换图，将观测统一到世界、基座或末端坐标系；同时统一关节顺序、力矩约定及传感器内外参。
5. **数据清洗与封装：**删除不完整轨迹，依据任务起止标记和人工注释裁剪空闲段；通过时间戳索引支持最近邻匹配或插值。最终以统一schema保存状态/动作、RGB-D、触觉、力/力矩、标定参数、任务与平台ID、成功标签和人工评分。
6. **训练验证：**将数据转为LeRobot v3.0格式，使用ACT、Diffusion Policy和π0进行预训练—微调。先在5000条轨迹上学习通用工业先验，再使用目标任务数据微调。

其设计重点不在模型结构，而在**数据质量、跨平台异构性和接触信息的联合组织**。

# 4. 方法对比分析

PRISM区别于Open X-Embodiment、RT-1等视觉主导数据集的关键，是把高精度工业约束和力/触觉信号置于核心位置；区别于REASSEMBLE等小规模接触数据集，则体现在更大的任务、机器人和遥操作覆盖。创新主要包括：统一多机器人、多末端、多遥操作接口的数据标准；保留高频力/力矩和原始时间戳；显式提供人类示范与机器人轨迹配对；支持从子任务到长流程的组合学习。

适合研究多模态融合、接触状态估计、工业模仿学习、跨机器人迁移和遥操作质量分析。但它并未提出新的闭环力控算法，且触觉仅覆盖部分样本。

# 5. 实验分析

作者在Realman双臂平台上测试电子插拔、卡尺包装和传送带分拣。主要结论是：增加任务示范数量、使用全数据预训练，通常能提高成功率和抗扰性；但现有策略在动态分拣和精密插拔上的表现仍明显不足，说明仅靠行为克隆难以解决时序预测与精确力调节。另一个重要发现是，外骨骼采集的数据普遍优于VR数据，表明遥操作接口会直接影响示范质量和最终策略性能。

优势是工业真实性、多模态同步和跨平台规模化；局限是成功率仍低、触觉覆盖不完整、实验环境与训练环境较一致，跨工厂泛化尚未充分验证。

# 6. 实用指南

论文提供数据集与项目页面（https://tengbo-yu.github.io/PRISM/）。复现时需重点处理多传感器时间对齐、坐标变换、关节顺序及力矩坐标系。训练中图像尺寸为540×960，动作chunk为15步，即15 Hz下约1秒；采用10%全数据预训练、90%任务微调。迁移到其他装配任务时，应保留统一schema，新增任务只需补充任务标签、成功标准和相应传感器标定；若迁移到新机器人，还需建立形态映射或采用末端位姿、相对动作等机器人无关表示。

# 7. 总结

**核心思想：**用规模化同步多模态数据支撑工业接触操作。

**速记版Pipeline：**

1. 用多机器人、多夹具和多种遥操作采集工业示范；  
2. 同步记录视觉、机器人状态、力/力矩及部分触觉；  
3. 完成清洗、标定、时间对齐和统一封装；  
4. 先用全数据学习通用操作先验，再针对任务微调；  
5. 在真实机器人上评估数据规模、预训练和采集方式的影响。

**Key Findings:**

- To address this gap, we introduce PRISM, a large-scale multimodal dataset for contact-rich industrial operations.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.17962v1)
- [arXiv](https://arxiv.org/abs/2608.17962v1)

---

<a id='2608.17874v1'></a>
## [Jetson-ORB-SLAM3: Accuracy-Preserving GPU Implementation for Edge Computing Devices](https://arxiv.org/abs/2608.17874v1)

**Authors:** Rajat Roy, Aditya Arun Kumar Yadav, Hardik Jain

**Published:** 2026-08-18

**Categories:** cs.RO

**Abstract:**

Visual-inertial SLAM on low-power edge platforms is constrained by the cost of dense feature extraction and loop closure. Prior GPU ports of ORB-SLAM trade accuracy for speed by approximating the ORB detector, altering the feature set and therefore the estimated trajectory. We present an accuracy-preserving GPU implementation of ORB-SLAM3 for the NVIDIA Jetson Orin Nano, whose GPU ORB front end reproduces the reference CPU detector algorithmically to 94.7% exact keypoint agreement and 99.9% descriptor bit agreement. This work also makes CNN-based loop closure edge-viable through native TensorRT. The visual front end (feature extraction) is offloaded to the GPU while the mapping and optimization back end is kept on the CPU, matching each computation to the hardware it suits. The accuracy is verified by comparing four configurations: the GPU pipeline and the unmodified CPU reference, each run on both the Jetson Orin Nano and a desktop. On EuRoC dataset, all four agree to within 0.10cm in mean absolute trajectory error (SE(3)), so neither the GPU port nor the change of hardware shifts the estimated trajectory. The GPU-versus-CPU comparison is reproducible on TUM-VI and KITTI datasets, so the acceleration is accuracy-preserving rather than approximate. The proposed implementation is competitive with published ORB-SLAM3 on EuRoC, attains sub-centimeter accuracy on five of the six TUM-VI room sequences, and reaches sub-1% relative translation error on nine of eleven KITTI sequences. For loop closure, the generic ONNX-Runtime CUDA/TensorRT execution providers are unusable with our CosPlace ResNet-50 on the embedded platform, whereas a native libnvinfer FP16 engine reduces per-query inference to 2.2ms, a 180x speedup. Learned place recognition therefore runs concurrently with tracking on a 7W device. In monocular-inertial mode the system sustains 32FPS mean over the eleven EuRoC sequences.

**Analysis:**

# 1. 摘要翻译

视觉—惯性SLAM在低功耗边缘设备上受到特征提取和回环检测开销的限制。已有GPU版ORB-SLAM常通过近似FAST检测器和非极大值抑制来换取速度，导致特征集合改变，进而影响轨迹估计。本文面向NVIDIA Jetson Orin Nano，提出一种保持精度的ORB-SLAM3 GPU实现：在GPU上逐算法复现CPU版ORB前端，关键点完全一致率达94.7%，描述子比特一致率达99.9%。同时，作者利用原生TensorRT FP16部署CosPlace ResNet-50，实现边缘设备上的CNN回环候选检索。GPU负责特征提取和地点识别，CPU保留跟踪、建图与优化后端。在EuRoC上，GPU/CPU及Orin/桌面四种配置的平均绝对轨迹误差相差不超过0.10 cm；在TUM-VI和KITTI上也保持近似轨迹精度。TensorRT将地点识别延迟从约396 ms降至2.2 ms，实现180倍加速。系统单目—惯性平均达到32 FPS，立体—惯性模式通过流水线重叠可达到相机帧率。

# 2. 方法动机分析

**驱动力：**在约7 W的Jetson设备上同时实现ORB-SLAM3的精度、实时跟踪和学习式回环检测。

**现有痛点：**GPU移植通常修改检测、NMS或特征选择规则，虽然速度提高，却改变数据关联和后端优化输入；传统DBoW2在视角变化、重复场景中易误检，而通用ONNX Runtime在Jetson上部署CosPlace存在初始化失败或CPU推理过慢的问题。

**核心假设：**只要GPU严格复现CPU前端的选择规则，特征差异就不会显著传播到轨迹；不同计算模块应按硬件特性分配，而不是盲目将整个SLAM搬到GPU。

# 3. 方法设计详解

## 3.1 总体流程

输入为单目/立体图像及IMU，输出为相机或机体位姿、局部地图和全局优化结果：

1. **尺度金字塔：**对图像逐层高斯模糊并降采样，采用可分离卷积CUDA核；每层特征数沿用CPU参考分配。
2. **FAST检测：**逐像素执行FAST-9，在低纹理网格单元使用较低阈值回退策略，保持单元占用率。
3. **多尺度NMS：**不采用“选最强角点”等近似方法，而是保留参考实现的网格均匀化、角点评分排序和筛选规则。
4. **方向计算：**利用关键点邻域的强度质心计算方向，并使用参考圆形掩膜。
5. **描述子生成：**将学习得到的BRIEF采样对按关键点方向旋转，生成256 bit steered rBRIEF描述子。
6. **CPU后端：**GPU返回关键点和描述子后，CPU继续执行匹配、位姿跟踪、IMU预积分、局部建图、局部BA、DBoW2回环验证及Atlas多地图优化。
7. **CNN回环检索：**每个关键帧灰度缩放至224×224，经过ImageNet归一化和CosPlace ResNet-50得到512维、L2归一化全局描述子。与历史关键帧做余弦相似度检索，排除最近20帧。
8. **自适应候选门控：**当已有至少10个分数时，阈值为  
   \[
   \tau=\mathrm{clip}(\mu+3\sigma,0.75,0.95)
   \]
   仅保留高于阈值的前5个候选，再交给DBoW2和ORB几何验证；至少10个内点才接受回环。

## 3.2 关键设计

CNN不直接替代几何验证，而是作为高召回粗检索器；DBoW2、ORB匹配和Sim(3)/SE(3)优化负责精确确认。部署上绕过ONNX Runtime，离线在目标设备构建TensorRT FP16 engine，运行时一次反序列化，并在独立CUDA流中异步执行，避免阻塞跟踪。

# 4. 方法对比与创新

**本质区别：**现有GPU ORB-SLAM追求“GPU友好”，本文追求“参考算法等价”；其创新不在提出新的SLAM后端，而在于将CPU前端的细节规则完整并行化，并用轨迹级实验验证等价性。

**主要贡献：**  
1. reference-faithful CUDA ORB前端；  
2. 面向Jetson的原生TensorRT学习式回环；  
3. 通过跨平台、跨数据集的2×2配置矩阵区分“移植误差”和“硬件差异”。

**适用场景：**低功耗机器人、无人机、室内巡检和GPS拒止环境；尤其适合已有ORB-SLAM3代码、但需要释放CPU或部署学习式地点识别的系统。

# 5. 实验分析

作者在EuRoC、TUM-VI和KITTI上比较GPU与CPU参考实现。代表性结论是：EuRoC四种配置平均ATE相差不超过0.10 cm，说明移植基本不改变轨迹；CosPlace TensorRT推理仅2.2 ms，且单目—惯性平均32 FPS，立体—惯性通过重叠执行达到28 FPS。

**优势：**精度保持、模块替换成本低、低功耗、学习式回环可实时运行。  
**局限：**EuRoC分辨率下GPU特征提取反而慢于多核CPU；性能依赖JetPack、CUDA和TensorRT版本；V203等强运动序列仍受IMU初始化和回环随机性影响；TensorRT engine不可直接跨设备复用。

# 6. 实用指南

论文提供代码仓库：`IITJ-CLARITY-Lab/Jetson-ORB-SLAM3`。复现时需固定JetPack 6.2、CUDA 12.6、TensorRT 10.3，使用二进制DBoW2词典，并在目标Jetson上重新构建约30 s的TensorRT engine。关键参数包括金字塔层级与特征分配、FAST阈值、候选数K=5、时间间隔20帧、最低相似度0.75、最高阈值0.95及10个几何内点。该思路可迁移到其他特征SLAM：保留原算法选择语义，优先并行其局部计算，再通过特征级和轨迹级一致性测试验证移植。

# 7. 总结

**核心思想：**严格复现前端，按硬件分配计算。

**速记版Pipeline：**

1. GPU按CPU规则构建图像层级并提取ORB。  
2. 保持检测、筛选、方向和描述子完全一致。  
3. CPU继续完成跟踪、建图和优化。  
4. TensorRT异步生成地点描述子并筛选回环候选。  
5. 通过ORB几何验证后执行全局校正。

**Key Findings:**

- We present an accuracy-preserving GPU implementation of ORB-SLAM3 for the NVIDIA Jetson Orin Nano, whose GPU ORB front end reproduces the reference CPU detector algorithmically to 94.7% exact keypoint agreement and 99.9% descriptor bit agreement.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.17874v1)
- [arXiv](https://arxiv.org/abs/2608.17874v1)

---

<a id='2608.17832v1'></a>
## [GenRec: Knowing Where to Reconstruct and Where to Generate](https://arxiv.org/abs/2608.17832v1)

**Authors:** Ata Çelen, Jaewoo Jung, Federico Tombari, Marc Pollefeys, Sunghwan Hong, Michael Niemeyer, Daniel Barath

**Published:** 2026-08-18

**Categories:** cs.CV

**Abstract:**

Generative novel view synthesis from sparse input images is rarely all reconstruction or all generation: pixels visible in some source view have a unique correct value modulated only by view-dependent shading, while pixels in disocclusions or beyond the captured volume admit a distribution of plausible completions. Existing generative novel-view-synthesis methods conflate these regimes under a single uniform loss, blurring the line between geometric fidelity and creative hallucinations even when scene geometry is injected through warped point clouds or projected depth. We introduce GenRec, a multi-view flow matching model that builds the reconstruction--generation split directly into its architecture, supervision, and gradient flow. Guided by an observation mask derived from the source cameras and a monocular depth estimator, a flow matching backbone jointly denoises RGB and scene-coordinate maps across all target views, while a pixel-space refinement stage restores high-frequency detail on observed pixels; the same mask gates supervision so regression signals do not contaminate the generative prior. Across RealEstate10K, DL3DV-10K, and Mip-NeRF~360, in both single-view extrapolation and two-view interpolation, GenRec attains the best reconstruction fidelity in observed regions while also surpassing purely generative baselines on perceptual quality in unobserved ones, showing the effectiveness of our approach.

**Analysis:**

## 1. 摘要翻译

从稀疏输入图像进行生成式新视角合成，很少是纯粹的“重建”或纯粹的“生成”：源视图中已经可见的像素具有唯一正确值，仅受视角相关光照影响；而遮挡后显露区域或相机覆盖范围之外的区域，只能从多个合理补全中进行选择。现有方法通常使用统一损失处理这两类区域，即使输入中注入了变形点云或投影深度，也混淆了几何保真与生成式幻觉。

本文提出 GenRec，一种多视图流匹配模型，在网络结构、监督方式和梯度传播层面显式划分“重建区域”和“生成区域”。模型利用源相机与单目深度估计器构造观测掩码；流匹配主干网络联合去噪 RGB 与场景坐标图，并覆盖所有目标视图；随后，像素空间细化模块仅在观测区域恢复高频细节。相同的观测掩码还用于限制监督，使回归信号不会污染生成先验。在 RealEstate10K、DL3DV-10K 和 Mip-NeRF 360 上，GenRec 在观测区域取得最佳重建保真度，同时在未观测区域超过纯生成基线，并显著提高推理速度。

## 2. 方法动机分析

**驱动力**：目标视图内部同时存在两种统计性质不同的像素。可观测区域应尽量复制真实外观，未观测区域则需要依赖生成先验。作者认为，统一生成损失必然造成折中。

**现有痛点**：NeRF/3DGS擅长插值但无法补全未观测区域；扩散或流匹配方法能生成合理内容，却把已观测像素也当作需要重新“猜测”的内容；基于warp-as-target的方法虽然几何一致，但会继承深度误差、孔洞和错位伪影。

**核心假设**：观测区域的答案近似确定，应采用像素级回归；未观测区域答案具有多模态不确定性，应保留生成模型的分布建模能力。两者必须在参数和梯度上解耦。

## 3. 方法设计详解

### Pipeline

1. **几何预处理**  
   对每个源图像估计单目深度，并利用已知相机位姿将源 RGB 和反投影三维点前向投影到每个目标相机。得到观测掩码 \(O_i\)、warp RGB \(W_i^I\) 和warp场景坐标 \(W_i^{SC}\)。掩码表示目标像素是否获得源视图证据。

2. **双模态流匹配主干**  
   每个目标视图同时生成 RGB latent 和 scene-coordinate latent。它们沿
   \[
   x_t=(1-t)x_0+tx_1
   \]
   从高斯噪声流向真实数据。主干初始化自 Stable Diffusion 2.1，并通过 Diff2Flow 将原有 v-prediction 转为流速度，从而使用较少Euler步采样。

3. **几何条件注入**  
   每个模态输入由噪声latent、目标视图Plücker射线、下采样观测掩码及对应warp编码拼接而成。网络加入两类注意力：跨视图注意力保证多个目标帧一致；零初始化的跨模态注意力让 RGB 与场景坐标联合推理。场景坐标不仅作为输出，也为后续三维对应关系提供几何基础。

4. **像素空间解码细化**  
   RGB latent经带 LoRA 的解码器恢复图像；同时利用零初始化的 skip convolution，把warp RGB的高频特征注入各个上采样阶段。该模块改善细节，但其损失通过掩码控制，避免未观测区域被强行复制或回归。

5. **稀疏三维重建分支**  
   解码图像、warp RGB、源图像和预测场景坐标共同输入重建分支。目标像素依据三维坐标检索源视图及其他目标视图中的近邻，仅对这些候选进行稀疏交叉注意力。距离权重以
   \[
   w_{n,k}\propto(d_{n,k}+\epsilon)^{-1}
   \]
   作为注意力先验，使几何上更近的对应点更受重视。随后通过跨目标视图注意力提高多帧一致性，输出残差 \(\Delta_i\)：
   \[
   \hat I_i^{rec}=\hat I_i+O_i\odot\Delta_i.
   \]
   因而未观测区域完全保留生成结果。

6. **解耦训练与梯度控制**  
   第一阶段训练流匹配主干；第二阶段冻结主干，仅训练解码适配器和重建分支。LPIPS 等全图损失通过 stop-gradient 处理未观测像素，且重建梯度不回传主干。作者还用多步Euler轨迹产生更接近测试时的latent，避免训练时单步估计与推理分布不一致。

## 4. 方法对比与创新

本质区别不是“加入了深度”，而是把**观测掩码变成结构、损失和梯度的共同控制信号**。主要创新包括：

- 生成主干负责全图分布建模，独立像素分支负责观测区域保真；
- RGB与场景坐标联合流匹配，提供可用于注意力检索的三维对应；
- 基于三维近邻的稀疏跨视图细化，而非全局昂贵注意力；
- 通过梯度隔离保护未观测区域的生成先验。

适合稀疏视图、较大视角外推、机器人和混合现实等场景；若目标几乎全未观测，方法退化为普通生成主干。

## 5. 实验分析

作者在三个数据集上测试单视图外推和双视图插值，并进行模块、深度鲁棒性及三维一致性消融。代表性结论是：GenRec在RE10K单视图外推中达到17.05 dB PSNR、0.6071 SSIM，并将推理时间控制在约11秒；在DL3DV-10K双视图插值中优于主流基线。消融显示，去除掩码门控或让重建梯度回传主干都会降低生成质量。

优势是重建与生成兼顾、速度快、可随观测覆盖率自适应。局限是依赖单目深度；掩码错误会导致错误区域被过度重建，且强烈视角相关效应和极低覆盖率仍较困难。

## 6. 实用指南

论文提供项目主页，但给定文本未明确说明完整代码和权重是否公开。复现关键点包括：SD2.1初始化、DA3深度与Umeyama尺度对齐、50步Euler采样、CFG=1.5、\(k=8\)三维近邻、RGB+场景坐标双模态训练，以及先训主干、再冻结主干训练重建分支。训练使用约16万场景、8帧窗口、1/2源视图混合，分辨率最长边616。迁移到视频修复、稀疏深度补全或机器人观测预测时，可将 \(O\) 替换为传感器置信度，并保留“确定区域回归、不确定区域生成”的分工。

## 7. 总结

**核心思想：按观测区域分离重建与生成。**

**速记版 Pipeline：**

1. 用深度把源图像投影到目标视角，标出哪些像素有证据。  
2. 用流模型同时生成目标图像和三维坐标。  
3. 只在有证据区域检索三维邻居并恢复细节。  
4. 用掩码限制修正范围和梯度，保护无证据区域的生成先验。

**Key Findings:**

- Generative novel view synthesis from sparse input images is rarely all reconstruction or all generation: pixels visible in some source view have a unique correct value modulated only by view-dependent shading, while pixels in disocclusions or beyond the captured volume admit a distribution of plausible completions.
- Existing generative novel-view-synthesis methods conflate these regimes under a single uniform loss, blurring the line between geometric fidelity and creative hallucinations even when scene geometry is injected through warped point clouds or projected depth.
- We introduce GenRec, a multi-view flow matching model that builds the reconstruction--generation split directly into its architecture, supervision, and gradient flow.
- Across RealEstate10K, DL3DV-10K, and Mip-NeRF~360, in both single-view extrapolation and two-view interpolation, GenRec attains the best reconstruction fidelity in observed regions while also surpassing purely generative baselines on perceptual quality in unobserved ones, showing the effectiveness of our approach.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.17832v1)
- [arXiv](https://arxiv.org/abs/2608.17832v1)

---

<a id='2608.17787v1'></a>
## [ETHEREAL: A 25.6-$μ$s/inf. Low-latency Event-driven Graph-neural-network Processor for High-resolution Vision at the Edge](https://arxiv.org/abs/2608.17787v1)

**Authors:** Adrian Kneip, Martin Lefebvre, Daniel Gehrig, Victoria Catalán Pastor, Davide Scaramuzza, Marian Verhelst, Charlotte Frenkel

**Published:** 2026-08-18

**Categories:** cs.AR, cs.CV

**Abstract:**

Dynamic vision sensors (DVS) are enticing candidates to reach the low-latency, sub-ms target of edge-vision applications, as they generate events with a $μ$s-level time resolution. However, using DVS front ends also calls for novel algorithm/hardware back ends capable of efficiently handling streams of sparse spatiotemporal events. While event-driven graph neural networks (EV-GNNs) have emerged as a solution on the algorithmic side that is both accurate and efficient, there is no dedicated hardware to date capable of efficiently supporting their mixed requirements of dense-regular compute operations and sparse-irregular memory accesses. We therefore introduce ETHEREAL, the first EV-GNN processor chip, capable of bridging this gap by means of a neighbor-parallel spline-convolution engine combined with a split-2D/3D memory hierarchy that introduces a novel spatiotemporal event-caching mechanism. Measurement results demonstrate a 25.6$μ$s latency and a 1.6$μ$J energy per end-to-end event-wise inference on the state-of-the art DAGr-GNN workload and VGA-resolution (640x480 pixels) DSEC dataset.

**Analysis:**

## 1. 摘要翻译

动态视觉传感器（DVS）能够以微秒级时间分辨率产生事件，非常适合低延迟、亚毫秒级的边缘视觉应用。然而，DVS产生的是稀疏的时空事件流，需要后端算法和硬件高效处理。事件驱动图神经网络（EV-GNN）在算法层面兼顾了精度与效率，但其同时包含规则而密集的计算操作，以及稀疏而不规则的存储访问，现有硬件难以高效支持。本文提出ETHEREAL，这是首款EV-GNN处理器芯片，通过邻居并行的样条卷积引擎，以及融合新型时空事件缓存机制的2D/3D分离式存储层次，解决上述问题。在DAGr-GNN和VGA分辨率（640×480）的DSEC数据集上，芯片实测端到端单事件推理延迟为25.6 μs，能耗为1.6 μJ。

## 2. 方法动机

**驱动力**：将DVS的微秒级感知能力真正转化为低延迟检测能力，而不是被后端处理拖慢。

**现有痛点**：  
1. GPU或同步GNN需要时间窗口聚合，牺牲事件流的异步特性；  
2. SNN虽低延迟，但高分辨率任务精度和训练复杂度存在问题；  
3. 既有EV-GNN FPGA设计只支持低分辨率，无法同时处理3D稀疏访存和2D高并行计算；  
4. 样条卷积需要动态索引、双重乘累加和位置相关系数，普通MAC阵列效率低；  
5. 3D图特征规模超过100 MB，而2D图虽可片上存储，却存在边索引访问瓶颈。

**核心假设**：EV-GNN的邻居连接虽然全局稀疏且不规则，但局部邻域具有可利用的结构规律；因此，应让计算阵列并行处理邻居，让3D存储利用时空局部性，让2D存储利用规则邻接性。

## 3. 方法设计详解

### 总体流程

输入事件为\((x,y,t)\)。首先，图构建模块在空间半径\(r_s\)和时间半径\(r_t\)内，以螺旋式搜索寻找最多16个历史邻居，并读取其节点特征。随后执行DAGr-GNN：

1. **3D图卷积**：在高分辨率时空图上处理少量通道特征；  
2. **3D到2D池化**：合并时间维度，将节点聚合到低分辨率体素；  
3. **多层2D图卷积与池化**：在规则的3×3邻域上扩大感受野；  
4. **YOLO式检测头**：输出当前事件触发的预测更新。

### 样条卷积引擎

作者将原始样条卷积重新排列为“先通道MAC、后样条调制”：

\[
Y_{MAC,n,j}=\sum_i X_{n,i}W_{i,j}[idx(\Delta pos_n)]
\]

再计算：

\[
Y_{msg,n,j}=\sum_d Z_d(\Delta pos_n)Y_{MAC,n,j}
\]

其中，\(idx\)表示由邻居相对位置选择的权重索引，\(Z_d\)是双线性插值系数。该重排的关键收益是：大规模通道MAC只需执行一次，而样条系数只作用于部分和，从而显著减少高精度操作。

硬件将消息传递分为三阶段：  
- **初始化**：根据邻居所在空间象限查表，合并有效样条索引；  
- **Linear-MAC**：对共享同一索引的邻居并行计算部分和；  
- **Spline-MAC**：将部分和与位置相关的\(Z\)系数相乘累加。

8个邻居并行处理核心分别承担不同邻居，并通过“样条跳过”跳过无效索引。每个核心支持4/8 bit输入和权重精度，可按层切换。之后统一聚合单元执行逐通道最大聚合、自线性分支、偏置、缩放、量化和ReLU。

### 分离式存储层次

- **3D时空缓存**：采用256组、8路空间关联结构，仅保留每个像素最近事件。命中时直接读取片上特征；未命中时访问外部存储，并通过写回机制维护数据。其本质是用事件的时空局部性减少外部存储访问。  
- **2D图Scratchpad**：将特征和相对位置交错映射到16个bank，并把8条入边编码为8 bit位图，避免逐邻居查找源—目的索引，实现无停顿的邻居并行流式读取。  
- **系统调度**：预存多层配置、硬件化池化，并将下一事件的3D图构建与当前事件的2D处理重叠，隐藏图构建延迟。

## 4. 对比与创新

其根本区别不是提出新的GNN结构，而是针对EV-GNN的“计算—存储耦合特性”进行软硬件协同设计。主要创新包括：  
1. 面向样条卷积的邻居并行、索引跳过数据流；  
2. 支持按操作数和网络层切换4/8 bit精度；  
3. 融合图构建的3D时空缓存；  
4. 利用规则邻接和bank交错的2D片上Scratchpad；  
5. 通过3D/2D流水调度降低端到端延迟。

适合高分辨率、事件异步、局部邻域更新的目标检测、机器人和自动驾驶场景。

## 5. 实验分析

作者在TSMC 28 nm、3.7 mm²芯片上验证DAGr-GNN。最具代表性的结果是：DSEC 640×480输入下，端到端延迟25.6 μs、能耗1.6 μJ/事件；3D缓存最多减少约60%的外部访存，2D存储相较串行方案最高获得57倍层延迟降低。

优势是低延迟、可扩展到高分辨率、精度—能耗可调。局限是3D缓存仍受命中率影响，邻居很少时并行核心利用率不足，且论文未覆盖完整CNN融合系统；部分位置池化仍由CPU处理。

## 6. 实用指南

文中未明确给出完整开源代码或RTL。复现需准备DSEC等事件数据、DAGr-GNN训练/QAT流程、异步图构建、4/8 bit量化、3D外存模拟器和2D邻接编码。关键参数包括\(r_s\)、\(r_t\)、最大邻居数16、3D缓存容量及各层通道精度。该框架可迁移到运动预测、姿态估计和事件分割，但需重新设计池化、检测头及图缓存策略。

## 7. 总结

**核心思想：用专用存储与并行卷积释放事件流低延迟。**

**速记版Pipeline**：  
1. 为每个新事件寻找局部时空邻居；  
2. 用缓存减少3D历史特征的外部读取；  
3. 按共享位置索引并行计算样条卷积；  
4. 将3D节点池化为2D体素并连续更新；  
5. 重叠下一事件建图与当前事件推理，输出检测结果。

**Key Findings:**

- However, using DVS front ends also calls for novel algorithm/hardware back ends capable of efficiently handling streams of sparse spatiotemporal events.
- We therefore introduce ETHEREAL, the first EV-GNN processor chip, capable of bridging this gap by means of a neighbor-parallel spline-convolution engine combined with a split-2D/3D memory hierarchy that introduces a novel spatiotemporal event-caching mechanism.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.17787v1)
- [arXiv](https://arxiv.org/abs/2608.17787v1)

---

<a id='2608.17779v1'></a>
## [Stability Control for Real World Testing in Autonomous Racing](https://arxiv.org/abs/2608.17779v1)

**Authors:** Phillip Pitschi, Simon Sagmeister, Frederik Werner, Markus Lienkamp, Boris Lohmann

**Published:** 2026-08-18

**Categories:** cs.RO, eess.SY

**Abstract:**

Controlling an autonomous vehicle at the limits of handling is a challenging task. Due to external influences, such as road conditions or weather, a vehicle can easily become unstable. Since most control algorithms assume stable vehicle behavior, they might fail in these situations. Especially when operating expensive vehicles without a safety driver on board, as in autonomous racing, this poses a significant challenge. To enable safe operation at the vehicle's dynamic limits, we present a comprehensive stability control system that safeguards motion control algorithms in autonomous driving. The proposed system consists of an electronic stability control (ESC), a slip control (SC), and a countersteer system (CS), which collectively adapt steering and brake commands from the motion controller to maintain vehicle stability. We validate our approach through both simulation and experiments on a real-world, full-scale vehicle. The results show that the stability control system maintains vehicle stability in critical situations and extends the operational feasible region. To simplify integration, we provide an open-source implementation at github.com/TUMFTM/tam-stability-control.

**Analysis:**

## 1. 摘要翻译

在操纵极限下控制自动驾驶车辆是一项具有挑战性的任务。受道路状况、天气等外部因素影响，车辆很容易失稳，而多数控制算法都假设车辆处于稳定状态，因此在这些场景下可能失效。尤其是在无人安全员、且车辆价值较高的自动驾驶赛车中，这一问题更加突出。为实现车辆在动态极限附近的安全运行，本文提出一种用于保护自动驾驶运动控制器的综合稳定性控制系统。该系统由电子稳定控制（ESC）、车轮滑移控制（SC）和反打方向控制（CS）组成，通过联合调整转向指令和制动指令维持车辆稳定。作者在仿真和全尺寸真实车辆上验证了该方法。结果表明，该系统能够在过度转向、车轮过度滑移等关键场景下保持车辆稳定，并扩大车辆的可运行区域。作者同时开源了C++实现。

## 2. 方法动机分析

**驱动力：**自动驾驶赛车需要在极限工况下测试控制器，但道路摩擦、载荷转移和扰动会使车辆响应偏离模型，导致失控、打滑甚至损坏轮胎。

**现有痛点：**传统运动控制器通常只负责轨迹跟踪，不专门处理失稳；已有SC或ESC多面向乘用车，通常未与自动驾驶控制器及主动反打方向协同；复杂MPC虽性能高，却计算负担大、对模型准确性敏感。

**核心假设：**运动控制器在正常工况下保持主导；一旦检测到失稳趋势，由一个低计算量、模块化的“安全覆盖层”临时修改转向和制动，而不是全面替代原控制器。并且，在过度转向时，转向修正比制动修正更快、对纵向性能损失更小。

## 3. 方法设计详解

### 总体Pipeline

1. **运动控制器输出：**生成转向角、各轮制动压力和发动机指令。发动机节气门不被稳定层修改，因制动压力响应更快。  
2. **状态估计：**获取车速、横摆角速度、侧偏角、各轮滑移率、前后轴滑移角及垂向载荷。  
3. **并行判断：**ESC处理横向横摆稳定性，SC处理纵向车轮滑移，CS处理过度转向下的转向修正。系统仅在阈值触发时介入。  
4. **指令融合：**输出修正后的转向角和单轮制动压力；正常情况下完全保留原运动控制器指令。

### ESC：基于横摆力矩的制动稳定控制

ESC仅在过度转向时工作，主要使用前轴左右制动器产生反向横摆力矩，避免制动后轴饱和轮胎进一步恶化。

- 根据期望曲率、车速和车辆参数计算参考横摆角速度与参考侧偏角：
  \[
  \dot\psi_{\rm ref}=\kappa v
  \]
  \[
  \beta_{\rm ref}=\kappa\left(l_r-\frac{l_fmv^2}{C_rl}\right)
  \]
- 曲率既可由转向角结合线性单轨模型得到，也可由目标横向加速度计算：
  \[
  \kappa=a_y/v^2
  \]
- 对横摆角速度误差和侧偏角误差低通滤波，再分别输入两个PID控制器，合成为目标横摆力矩。
- 下层压力分配器通过一侧减压、另一侧增压实现目标力矩。只有当误差、绝对侧偏角和车速同时超过动态阈值时才激活；阈值随车速调整，以避免低速高曲率弯道中的误触发。

### SC：无模型有限状态机

SC不是连续模型控制，而是针对每个车轮的规则状态机，状态包括**保持、增加、降低**。其关键操作是调整制动压力比例 \(\epsilon\)。

- 当滑移率超过激活阈值 \(\lambda_{ih}\) 时进入保持状态，并保存当前制动压力作为目标基准。
- 根据滑移率 \(\lambda\) 及其变化率 \(\dot\lambda\) 在三个状态间切换，以补偿制动执行器延迟。
- 当滑移不足且低于安全阈值时，允许\(\epsilon>1\)，防止传感器噪声或错误触发造成不必要的制动力下降。
- 减速时各轮独立控制；加速时后轴整体控制，因为赛车的锁止差速器会将左右驱动力耦合，单独制动会产生非期望横摆力矩。
- 减速过程中逐步降低目标压力，以适应速度下降导致的空气下压力和垂向载荷减少。

### CS：基于前后轴滑移角差的反打方向

CS是本文最有代表性的设计。其不是直接追踪复杂非线性轮胎模型，而是利用前后轴滑移角关系判断过度转向。

车辆稳态转弯满足：
\[
\delta=\dot\psi l/v+\alpha_f-\alpha_r
\]
当后轴滑移角超过前轴，即 \(\alpha_r>\alpha_f\)，说明后轴更接近饱和、车辆出现过度转向。此时计算：
\[
\Delta\delta=\alpha_r-\alpha_f
\]
并将运动控制器转向指令修改为：
\[
\bar\delta=\delta-\Delta\delta
\]
即减小转向角，使车辆回到中性或轻微不足转向状态。前后轴滑移角由左右车轮滑移角按垂向载荷加权平均，从而考虑侧向载荷转移。CS阈值比ESC更敏感，因为转向执行器延迟约30 ms，而制动约150 ms。

## 4. 方法对比与创新

**本质区别：**该方法不是重新设计一个完整运动控制器，而是作为“稳定性安全覆盖层”与任意运动控制器解耦；同时将制动滑移、横摆稳定和反打方向统一到同一架构中。相比依赖精确车辆模型的MPC，SC采用模型无关状态机，CS则有意采用线性、小角度近似以增强模型失配鲁棒性。

**创新贡献：**  
1. ESC、SC、CS的模块化协同设计；  
2. 通过前后轴滑移角差直接触发反打方向；  
3. 面向真实赛车的低延迟、低计算量实现；  
4. 允许运动控制器更安全地探索极限性能，而非在达到极限后突然失控。

**适用场景：**高性能车辆、自动驾驶赛车、摩擦变化明显或无安全员的真实道路测试。对严重不足转向、需要持续横向动力分配的场景支持有限。

## 5. 实验分析

作者使用双轨Pacejka模型进行仿真，并与Pure Pursuit、点质量Tube-MPC和单轨NMPC结合；随后在Dallara EAV25赛车、Yas Marina赛道进行约40小时实测。

关键结论：  
- SC显著抑制车轮抱死和空转，使车辆可承受更高纵向加速度限制。  
- ESC与CS联合时，车辆在降低后轴抓地力的过度转向仿真中仍能稳定运行；实测期间无碰撞、无自旋和轮胎损坏。

优势是响应快、可解释、计算开销低、易插入现有控制器。局限是阈值依赖人工调参，ESC参考模型基于线性轮胎，且系统主要处理过度转向，无法根本提升底层运动控制器能力。

## 6. 实用指南

论文提供开源C++实现：`github.com/TUMFTM/tam-stability-control`。复现时需搭建车辆状态估计、轮速/滑移角计算、轮胎载荷估计和单轮制动接口，并分别标定ESC/CS触发阈值、PID增益、SC滑移阈值及压力比例。作者未采用神经网络训练，重点是执行器延迟、滤波器和阈值随速度变化的实现。迁移到其他车辆时，应重新辨识轴距、质心位置、轮胎刚度、差速器形式及制动/转向延迟；若车辆具备电机独立驱动，还可将ESC扩展为直接横摆力矩或扭矩矢量控制。

## 7. 总结

**核心思想：**用快速安全覆盖层保护极限驾驶。

**速记版Pipeline：**  
1. 读取车速、横摆、侧偏和车轮滑移状态；  
2. 判断是否出现过度转向或车轮滑移；  
3. 用前后轴滑移差减小转向角；  
4. 用单轮制动产生反向横摆力矩并抑制滑移；  
5. 稳定后退出干预，将控制权还给原运动控制器。

**Key Findings:**

- To enable safe operation at the vehicle's dynamic limits, we present a comprehensive stability control system that safeguards motion control algorithms in autonomous driving.
- We validate our approach through both simulation and experiments on a real-world, full-scale vehicle.
- The results show that the stability control system maintains vehicle stability in critical situations and extends the operational feasible region.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.17779v1)
- [arXiv](https://arxiv.org/abs/2608.17779v1)

---

