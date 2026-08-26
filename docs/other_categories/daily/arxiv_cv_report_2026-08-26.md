time: 20260826

# Arxiv Computer Vision Papers - 2026-08-26

## Executive Summary

# ArXiv 计算机视觉日报执行摘要  
**发布日期：2026 年 8 月 25 日｜论文数量：10 篇**

> 注：以下判断主要依据论文标题、研究方向与作者提供的题录信息，适合作为快速筛选摘要；具体方法、实验结果与结论仍应以论文正文为准。

## 1. 主要主题与趋势

### 1）视频数据与多模态基础模型基础设施持续扩张
- **LAION-BVD**提出千万小时级开放视频数据集，体现出视觉—语言—视频预训练正从“高质量小规模数据”转向“超大规模、开放、可复用的数据基础设施”。
- **WeMM-Embedding**聚焦微信多模态嵌入技术，说明工业界正在加强面向真实应用场景的统一多模态表示学习，尤其关注文本、图像及可能的短视频/社交内容之间的检索与匹配能力。

### 2）世界模型研究从“生成未来”转向“验证动作因果性”
本期有多篇论文集中讨论世界模型与动作建模：
- **Do Robotic World Models Really Follow Actions?**直接质疑机器人世界模型是否真正遵循输入动作，而不是仅生成视觉上合理的未来。
- **Latent Action as Intention**尝试将潜在动作解释为“意图”，以提升未来想象和动作规划效率。
- **Game2World Engine**利用野外游戏视频训练世界模型，探索从非机器人、非实验室视频中获取交互和动态知识。

整体趋势是：世界模型研究正在从单纯的视频预测，转向**动作条件生成、因果一致性、可控性与策略学习有效性**。

### 3）机器人策略学习更加关注可执行性与现实约束
- **VIP**针对机器人导航中的迭代式规划与变化环境。
- **Gripper-aware VLA**将夹爪状态或夹爪几何纳入视觉—语言—动作模型，强调末端执行器感知对操作成功率的重要性。
- **NeuralParker**面向不规则停车环境设计强化学习规划器，体现了从标准化仿真任务走向复杂、非规则现实场景的趋势。

这些工作共同表明，机器人智能正从“理解场景”进一步转向**理解自身执行能力、动作约束和环境变化**。

### 4）高效模型适配与轻量化部署
- **Low-Rank Ternary Adaptation**结合低秩结构与三值参数，目标是降低Transformer微调的存储和计算成本。
- **MaST**通过运动感知稀疏管线实现轻量级目标跟踪，面向实时视觉应用中的效率问题。

这反映出视觉模型研究的另一条主线：在模型规模持续增长的同时，研究重点转向**低成本微调、边缘部署和实时推理**。

---

## 2. 值得特别关注的论文

### **LAION-BVD**
如果数据集规模、开放性和许可条件达到题目所暗示的水平，这可能是本期最具基础设施意义的工作之一。千万小时级视频数据将影响：
- 视频—语言预训练；
- 世界模型和视频生成；
- 时序动作理解；
- 数据筛选、去重、版权与质量评估方法。

其长期价值很大程度上取决于数据质量、元数据、过滤流程、许可合规性以及是否提供高效的检索和数据使用工具。

### **Do Robotic World Models Really Follow Actions?**
这篇论文的问题意识尤其重要。许多世界模型可能能够生成“看起来合理”的未来，但并不意味着它们准确建模了动作对环境的因果影响。若论文提出了可靠的诊断指标、反事实测试或对齐方法，将对以下方向产生直接影响：
- 基于世界模型的策略学习；
- 机器人视频预测；
- 模型预测控制；
- 动作条件视频生成的评测体系。

### **Latent Action as Intention**
将潜在动作建模为高层意图，可能有助于缓解原始动作空间过大、动作序列过长以及未来想象成本高等问题。该思路连接了：
- 表征学习；
- 分层控制；
- 逆动力学；
- 目标条件规划；
- 世界模型中的抽象动作建模。

### **Game2World Engine**
从游戏视频中学习世界模型具有较强的规模化潜力。游戏环境通常包含丰富的交互、状态变化和行动结果，但与机器人现实之间存在明显的形态和动力学差异。因此，该工作的关键价值在于其是否解决了：
- 从视频中识别可操作事件；
- 从无动作标签视频中恢复动作；
- 游戏知识向现实机器人迁移；
- 游戏动态与真实物理之间的域差异。

### **Gripper-aware Vision Language Action Models**
将夹爪纳入VLA模型是一个具有现实意义的细化方向。相比只关注“目标物体”，机器人操作还必须考虑夹爪姿态、可达性、碰撞和抓取稳定性。该工作若能证明夹爪感知显著提升泛化或执行成功率，可能推动VLA模型从语义驱动走向更完整的**本体感知与执行感知**。

---

## 3. 正在形成的研究方向与技术

1. **动作可验证的世界模型**  
   未来评测不应只看视频预测质量，还应检验模型是否满足动作—结果之间的因果关系，并支持反事实动作测试。

2. **潜在动作、意图和分层策略**  
   将低层控制序列压缩为高层潜在意图，有望降低规划复杂度，并提升跨任务、跨机器人迁移能力。

3. **从非机器人视频学习交互模型**  
   游戏视频、网络视频和大规模公开视频可能成为训练世界模型的重要来源，但需要更强的动作恢复、时序理解和域适配技术。

4. **执行器感知的多模态机器人模型**  
   VLA模型将不仅理解语言和场景，也会显式建模夹爪、机械臂状态、可达空间和接触过程。

5. **面向复杂环境的强化学习规划**  
   导航和停车等任务正在从规则、结构化环境转向不规则、动态和现实约束更强的场景。

6. **低比特、低秩和稀疏化模型适配**  
   低秩三值适配与稀疏跟踪管线代表了降低训练和推理成本的两条路径，适合边缘设备和大规模部署。

7. **开放视频数据集的质量与治理**  
   数据规模之外，去重、质量分级、版权合规、偏差控制和可追溯性将成为视频基础模型数据集的核心竞争力。

---

## 4. 建议优先阅读全文的论文

### 第一优先级：世界模型与机器人学习
1. **Do Robotic World Models Really Follow Actions?**  
   优先了解其诊断方法、评测协议和动作对齐机制。该问题具有较强的基础性和普适性。

2. **Latent Action as Intention Enables Efficient Future Imagination for World Action Models**  
   适合关注世界模型、动作抽象和高效规划的研究人员，可能提供一种降低想象成本的新建模范式。

3. **Gripper-aware Vision Language Action Models**  
   对机器人操作、VLA和具身智能研究具有直接参考价值，重点查看其夹爪表示方式和真实执行实验。

### 第二优先级：数据与可扩展训练
4. **LAION-BVD**  
   对视频预训练、数据工程和世界模型研究者价值最高。建议重点阅读数据构建、筛选、许可、基准和数据效率分析。

5. **Game2World Engine**  
   适合关注从互联网视频或游戏数据训练世界模型、以及模拟到现实迁移的研究人员。

6. **WeMM-Embedding**  
   对多模态检索、工业级嵌入模型和大规模应用部署较为相关，建议重点评估其训练数据、任务覆盖和与现有嵌入模型的比较。

### 第三优先级：规划与高效部署
7. **VIP: Variation-based Iterative-learning Planning for Robotic Navigation**  
   适合导航、在线规划和动态环境决策方向。

8. **NeuralParker**  
   适合研究强化学习规划、自动驾驶或复杂停车场景的读者。

9. **Low-Rank Ternary Adaptation for Fine-Tuning Transformers**  
   对大模型压缩、参数高效微调和低资源部署具有潜在价值。

10. **MaST: Motion-aware Sparse Pipeline for Lightweight Object Tracking**  
   若研究重点是实时跟踪、移动端视觉或边缘计算，该论文的优先级可上调。

## 总结

本期论文的核心信号是：计算机视觉正加速与**世界模型、机器人控制、基础模型数据工程及高效部署**融合。最值得关注的转变并非单纯提升视觉生成质量，而是要求模型真正理解“执行某个动作会带来什么后果”，并能在现实约束下高效规划和行动。就研究影响力而言，**LAION-BVD、动作对齐世界模型、潜在意图动作建模以及夹爪感知VLA**构成了本期最值得优先跟进的四个方向。

---

## Table of Contents

1. [LAION-BVD: A 10-Million-Hour Open Video Dataset for Multimodal Pre-training](#2608.24845v1)
2. [WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report](#2608.24053v1)
3. [Do Robotic World Models Really Follow Actions? Diagnosing and Aligning Action-Conditioned Generation for Policy Learning](#2608.24885v1)
4. [Latent Action as Intention Enables Efficient Future Imagination for World Action Models](#2608.24882v1)
5. [Game2World Engine: Unlocking In-the-Wild Gameplay Videos for World Model Training](#2608.24680v1)
6. [VIP: Variation-based Iterative-learning Planning for Robotic Navigation](#2608.24618v1)
7. [Gripper-aware Vision Language Action Models](#2608.24603v1)
8. [NeuralParker: A Reinforcement Learning Planner for Irregular Parking Environments](#2608.24485v1)
9. [Low-Rank Ternary Adaptation for Fine-Tuning Transformers](#2608.24469v1)
10. [MaST: Motion-aware Sparse Pipeline for Lightweight Object Tracking](#2608.24365v1)

---

## Papers

<a id='2608.24845v1'></a>
## [LAION-BVD: A 10-Million-Hour Open Video Dataset for Multimodal Pre-training](https://arxiv.org/abs/2608.24845v1)

**Authors:** Andreas Hochlehnert, Marianna Nezhurina, Mehdi Cherti, Andrej Radonjic, Thaddäus Wiedemer, Christoph Schuhmann, Romain Beaumont, Wieland Brendel, Bernhard Schölkopf, A. Sophia Koepke, Jenia Jitsev, Matthias Bethge

**Published:** 2026-08-25

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

We present LAION-BVD, a large-scale open video dataset for multimodal learning, which contains 1.3B platform-specific video URLs collected from CommonCrawl. From these, we download 80M videos with a total duration of 10 million hours. The dataset is designed for multimodal pre-training across the video, audio, and image modalities. Using content-aware scene detection, we extract clips for which we synthetically generate video and audio captions. Models trained on these data achieve competitive performance on standard video-text and audio-text benchmarks, with consistent improvements as training or model scale increases. Additionally, we explore video frames as an alternative source of image-text data by extracting scene-changing frames. These frames exhibit a visual distribution distinct from standard web image corpora, and models trained on this dataset achieve strong image-text retrieval performance. We release LAION-BVD to the research community. It significantly expands open access to multimodal videos at an unprecedented scale.

**Analysis:**

# 1. 摘要翻译

本文提出 **LAION-BVD（LAION Big Video Dataset）**，一个面向多模态学习的大规模开放视频数据集。作者从 CommonCrawl 收集了 **13亿个平台视频URL**，成功下载约 **8000万段视频**，总时长达 **1000万小时**。数据集同时支持视频、音频和图像模态训练。作者利用内容感知的场景检测将视频切分为片段，并自动生成视频和音频描述；同时提取场景变化帧并生成图像描述。基于这些数据训练的模型在视频-文本、音频-文本基准上取得有竞争力的结果，并且随着数据量和模型规模增加而持续提升。视频帧还构成了区别于传统网页图像语料的视觉分布，在图像-文本检索任务上表现良好。作者向研究社区开放该数据集及相关元数据。

# 2. 方法动机分析

**驱动力**：现有开放视频数据集远小于图像和文本数据集，限制了视频、音频多模态基础模型的规模化训练。真正瓶颈不是视频存在性，而是下载、切分、筛选和标注的工程成本。

**现有痛点**：主流视频数据集规模通常只有百万级视频或几十万小时；不少数据依赖ASR、字幕或人工标注，模态覆盖不完整；视频帧通常被简单当作网页图像，无法体现其独特视觉分布。

**核心假设**：即使仅进行轻量筛选，并使用自动生成的短描述，只要视频规模足够大、场景切分合理，仍能提供有效且可扩展的多模态训练信号。

# 3. 方法设计详解

## 整体Pipeline

1. **URL构建**：处理截至2024年3月的CommonCrawl WAT文件，用`yt-dlp`提取平台视频链接，并借助`cc2dataset + Apache Spark`进行分布式处理。从47亿候选URL中过滤出YouTube、Vimeo和Dailymotion链接，得到13亿视频URL。  
2. **视频下载**：使用约2000台虚拟服务器、Celery集群、`yt-dlp`及住宅代理网络下载；尝试下载1.3亿视频，成功约8000万段，总时长1000万小时。  
3. **视频/音频片段生成**：随机抽取240万视频，去除短于10秒或长于30分钟的视频；使用PySceneDetect进行场景检测，阈值为30；再依据低分辨率帧间运动估计，删除基本静止的片段，形成约5500万场景片段。片段视频和对应音频保持时间对齐。  
4. **视频描述**：每个片段均匀采样最多32帧，输入Qwen3-VL-2B-Instruct，提示词为“用20词以内描述视频”，得到视频-文本对。  
5. **音频描述**：用Audio Flamingo 3处理对应音频，提示“用10词以内描述声音”，得到音频-文本对。  
6. **帧数据构建**：从更大视频池提取关键帧，删除黑帧，并用FFmpeg场景变化检测（阈值0.1）保留转场/场景变化帧，最终得到3亿帧；使用DeepSeek-VL2-tiny重新生成帧描述。

## 模型协同

作者并未提出新的编码器，而是将数据用于三类对比学习模型：ViCLIP负责视频-文本对齐，CLAP负责音频-文本对齐，CLIP负责帧-文本对齐。ViCLIP对8帧进行时空建模；CLAP将音频和文本映射到共享空间；CLIP将关键帧与描述进行对比学习。训练目标本质上是InfoNCE：匹配样本相似度高，不匹配样本相似度低。

# 4. 方法对比与创新

本质区别在于：LAION-BVD不是依赖人工标签或单一ASR监督，而是建立了一个**从网页视频发现、规模化下载、场景切分到视频/音频/帧独立标注的统一流水线**。主要创新包括：

- 将开放网页视频扩展至1000万小时；
- 用场景变化而非固定时间窗口生成训练片段；
- 同时构造视频、音频和视频帧三种监督；
- 强调视频帧具有区别于网页图像的分布，可作为互补图像数据源。

最适合视频-文本检索、音频检索、跨模态表示学习和大规模预训练；不适合直接替代高质量人工标注数据进行精细分类或生成式任务。

# 5. 实验分析

作者分别训练ViCLIP、CLAP和CLIP进行验证。代表性结论是：  
- LAION-BVD训练的ViCLIP在视频分类与检索上优于匹配规模的InternVid模型，且数据量、模型规模增加时性能总体提升。  
- 3亿帧训练的CLIP在MS-COCO、Flickr30k检索上很强，但ImageNet零样本分类较弱，原因是合成描述更像冗长的COCO式描述，且ImageNet类别词覆盖量明显低于DataComp。

优势是规模大、模态丰富、扩展性强；局限是自动描述存在错误，过滤较少导致偏见和有害内容风险，且尚未验证联合音视频模型或生成式模型。

# 6. 实用指南

论文提供URL、部分标注数据和HuggingFace资源；原始视频需研究机构接受条款后下载。复现重点是分布式URL解析与下载、场景检测、静态片段过滤，以及三种captioner的批量推理。关键设置包括：视频10秒—30分钟、最多32帧、视频描述20词以内、音频描述10词以内、帧检测阈值0.1。迁移到其他任务时，可替换captioner或加入ASR、OCR、说话人信息，并在时间轴上保留多模态同步关系。

# 7. 总结

**核心思想：用场景切分扩展开放视频多模态监督。**

**速记版Pipeline：**
1. 从网页抓取并下载大规模视频；  
2. 按场景切片并删除静止片段；  
3. 分别生成视频描述和声音描述；  
4. 提取场景变化帧并生成图像描述；  
5. 用三类对比模型验证数据价值。

**Key Findings:**

- We present LAION-BVD, a large-scale open video dataset for multimodal learning, which contains 1.3B platform-specific video URLs collected from CommonCrawl.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24845v1)
- [arXiv](https://arxiv.org/abs/2608.24845v1)

---

<a id='2608.24053v1'></a>
## [WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report](https://arxiv.org/abs/2608.24053v1)

**Authors:** Junjie Zhou, Ke Mei, Lei Li, Tianyi Wang, Fengyun Rao, Jing Lyu

**Published:** 2026-08-25

**Categories:** cs.CV, cs.CL, cs.IR

**Abstract:**

Universal multimodal embeddings are becoming a core component of modern AI systems, enabling heterogeneous content to be represented in a shared space for applications such as retrieval, recommendation, classification, and agentic systems. In this report, we present WeMM-Embedding, a family of universal multimodal embedding models supporting text, images, videos, visual documents, and arbitrarily interleaved multimodal inputs with flexible output dimensions. The family comprises 2B, 4B, and 9B variants and is trained in two stages: a large-scale multimodal alignment stage, followed by a refinement stage using curated data, fine-grained relevance supervision, and cross-scale knowledge transfer. Across extensive evaluations, WeMM-Embedding achieves leading performance on multiple public benchmarks. Notably, the 2B variant already surpasses the previously leading 8B open-source baseline on MMEB-v2, while the 9B variant further achieves a new state-of-the-art overall score of 80.6. WeMM-Embedding also demonstrates strong practical performance across WeChat applications, with substantial gains on a 26-task in-house benchmark and consistent improvements across 14 online A/B tests. It has been deployed at scale across recommendation and search applications, including WeChat Channels, Official Accounts, Moments, and e-commerce services. We have released the model weights and code to facilitate future research at https://github.com/Tencent/WeMM-Embedding.

**Analysis:**

## 1. 摘要翻译

通用多模态嵌入正成为现代 AI 系统的核心组件，可将文本、图像、视频、视觉文档及任意交错的多模态输入映射到共享空间，用于检索、推荐、分类和智能体系统。本文提出 **WeMM-Embedding**，包含 2B、4B 和 9B 三种规模。模型采用两阶段训练：首先进行大规模多模态对齐，随后利用精选数据、细粒度相关性监督和跨规模知识迁移进行精炼。在多个公开基准上，2B 模型已超过此前领先的 8B 开源基线，9B 模型在 MMEB-v2 上取得 80.6 的新 SOTA。模型还在微信内部 26 项任务和 14 次线上 A/B 测试中表现出稳定收益，并已部署于视频号、公众号、朋友圈和电商等推荐与搜索场景。

## 2. 方法动机分析

**驱动力与痛点：** 传统 CLIP/双塔模型通常为不同模态设计独立编码路径，难以统一处理“文本+图片”“视频+ASR 转写”或复杂组合查询。将视觉编码器接入文本模型虽扩大了适用范围，但对任意交错输入、细粒度相关性和多任务共享仍有限。作者的核心目标是构建一个既支持多模态组合、又能高效服务检索与推荐的统一嵌入模型。

**核心假设：** 先用大规模、宽覆盖数据建立统一语义空间，再通过高质量数据、困难负样本、排序监督和教师蒸馏修正局部相似度结构，能够同时获得通用性、判别性和参数效率。

## 3. 方法设计详解

### 3.1 统一数据表达

所有任务被转化为统一的源—目标匹配样本：

\[
z=(I,q,c,N,y)
\]

其中 \(I\) 是任务指令，\(q\) 为查询或源输入，\(c\) 为正目标，\(N\) 为显式困难负样本，\(y\) 为连续或离散相关性分数。源和目标均可包含文本、图片、视频及其交错组合，因此分类、问答、检索和推荐可以进入同一训练框架。

数据包括弱监督图文/视频文本对、细粒度描述、跨模态检索对、分类对、多模态问答和分级相关性样本。精选数据约为原始大规模数据的十分之一：先用中间模型编码较长一侧，并通过三级残差 K-Means 获得 Semantic ID；高密度语义区域降采样，稀疏区域提高保留率，以缓解数据分布偏斜。之后用多模态大模型进行质量过滤和文本纠错，并构造困难负样本：文本负样本由模型生成，图像/视频负样本由中间嵌入模型检索，部分候选再由 reranker 打分。

### 3.2 模型结构与推理

模型基于原生多模态 Qwen3.5，使用统一主干处理文本、图像和视频。输入序列保留原始模态顺序，并在末尾追加专用 `<embedding>` token。模型采用因果注意力，最后取该 token 的隐藏状态并 L2 归一化作为嵌入。

该设计还支持在序列中插入多个 `<embedding>` token。例如视频后放置一个 token、ASR 文本后再放置一个 token，即可一次前向同时得到“视频独立表示”和“视频—文本联合表示”。

通过 Matryoshka Representation Learning，仅保留隐藏向量前 \(d\) 个维度并重新归一化，即可从一次前向获得多种维度表示，适合不同存储和检索成本约束。

### 3.3 两阶段训练

**阶段一：大规模对齐。** 标准配对数据使用 InfoNCE，对显式困难负样本和批内负样本共同优化。作者的关键修正是：

1. **任务一致批处理：** 每个 batch 来自同一数据源和候选空间，使批内负样本更可靠；
2. **双侧重复感知掩码：** 若源或目标与当前样本近重复，则不将其作为负样本，降低分类重复标签导致的 false negative；
3. **分级相关性学习：** 对带有相关性等级的数据使用 score-gap 加权 CoSENT，使高相关样本相似度高于低相关样本，且相关性差距越大，约束越强；
4. **多维联合优化：** 在所有支持的嵌入维度上分别计算损失并加权求和。

**阶段二：精选微调与蒸馏。** 延续对比学习和分级排序，同时引入两类监督：

- **Reranker 监督：** 对同一查询下的正样本和困难负样本按 reranker 分数建立排序约束，仅在确实有效的任务子集使用，避免无差别引入噪声；
- **嵌入蒸馏：** 9B 教师分别计算源到目标、目标到源的相似度分布，学生通过双向 KL 散度拟合教师的软分布。相比 one-hot 对比目标，它保留了候选之间的相对相似度结构，尤其有利于 2B、4B 小模型。

2B 和 4B 使用冻结的 9B 作为教师；9B 没有更大教师，因此训练多个数据配置的版本后进行模型合并。

## 4. 方法对比与适用性

其本质区别不只是“把视觉输入接入 LLM”，而是将**任意交错输入、统一任务格式、重复感知负样本、分级排序、多维表示和跨规模蒸馏**组合成一个训练体系。创新主要集中在数据与监督组织方式，而非单一网络结构。它适合跨模态搜索、视频/文档检索、推荐召回、长尾内容匹配及智能体记忆检索；若任务需要音频，目前模型不支持。

## 5. 实验分析

作者在 MMEB-v2/v3、12 个跨模态检索数据集和微信 26 任务集上评测，并进行消融与线上 A/B 测试。代表性结论是：2B 在 MMEB-v2 达到 77.9，超过此前 8B 开源基线；9B 达到 80.6。MRL 表明 256 或 512 维即可保留大部分性能。局限包括：依赖大规模精选数据、reranker 和教师模型，训练成本高；实验中音频任务得分为零；部分线上收益和数据细节未充分公开，复现存在障碍。

## 6. 实用指南

模型权重和代码已开源于 Hugging Face 与 GitHub。复现重点是：保持任务一致 batch；实现源、目标双侧重复掩码；同时训练多个 MRL 维度；为困难负样本建立任务内候选池；仅在可靠任务上使用 reranker；用 9B 教师生成双向温度化相似度分布。迁移到新领域时，应将任务改写为“查询—候选”形式，补充领域正负样本和等级相关性标注，并重新构造 Semantic ID、候选池及指令模板。

## 7. 总结

**核心思想：** 以统一匹配训练多模态嵌入。

**速记版 Pipeline：**

1. 把不同任务统一成“输入—候选”的匹配样本。  
2. 用多模态主干和专用标记生成统一向量。  
3. 先用海量数据建立宽泛语义空间。  
4. 再用精选数据、难负样本、排序和教师软目标精修。  
5. 通过嵌套维度输出适配不同部署成本。

**Key Findings:**

- In this report, we present WeMM-Embedding, a family of universal multimodal embedding models supporting text, images, videos, visual documents, and arbitrarily interleaved multimodal inputs with flexible output dimensions.
- Notably, the 2B variant already surpasses the previously leading 8B open-source baseline on MMEB-v2, while the 9B variant further achieves a new state-of-the-art overall score of 80.6. WeMM-Embedding also demonstrates strong practical performance across WeChat applications, with substantial gains on a 26-task in-house benchmark and consistent improvements across 14 online A/B tests.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24053v1)
- [arXiv](https://arxiv.org/abs/2608.24053v1)

---

<a id='2608.24885v1'></a>
## [Do Robotic World Models Really Follow Actions? Diagnosing and Aligning Action-Conditioned Generation for Policy Learning](https://arxiv.org/abs/2608.24885v1)

**Authors:** Sixiang Chen, Jiaming Liu, Jixian Wu, Yichen Guo, Tinghao Wang, Siyuan Qian, Hao Chen, Jiajun Cao, Jian Tang, Shanghang Zhang

**Published:** 2026-08-25

**Categories:** cs.RO, cs.CV

**Abstract:**

Action-conditioned world models are increasingly used as learned simulators for policy evaluation and improvement, yet their effectiveness rests on an unverified assumption: generated futures faithfully reflect arbitrary valid actions. Existing benchmarks are typically confined to expert demonstrations, leaving off-expert action following inadequately evaluated. To address this gap, we introduce WorldEcho, which probes action following over a broader action distribution using visual integrity and SE(3) trajectory alignment. Our diagnosis shows that current world models reasonably execute expert actions but struggle with diverse off-expert trajectories, either ignoring the commanded actions or producing visually invalid rollouts. We further propose WorldSync, which strengthens action following along three complementary axes: distributional coverage, representational grounding, and intervention-effect alignment. It broadens the training distribution over action consequences, grounds intermediate video representations in action-induced robot dynamics through an Action-Forcing Expert, and aligns predicted changes under action interventions with the corresponding changes in ground-truth futures. Experiments on RoboTwin benchmarks and real-robot tasks show that WorldSync improves WorldEcho metrics and serves as a more reliable simulator for iterative policy improvement, enabling policies to achieve higher success rates.

**Analysis:**

# 1. 摘要翻译

动作条件世界模型常被用作策略评估与改进的学习式模拟器，但其有效性依赖一个尚未充分验证的假设：模型生成的未来能够忠实反映任意合法动作。现有基准主要局限于专家示范动作，对专家分布之外动作的跟随能力评估不足。

为此，本文提出 **WorldEcho**，通过视觉完整性与端执行器 SE(3) 轨迹对齐，从更广泛的动作分布评估动作跟随能力。实验发现，当前世界模型能够较好地执行专家动作，但面对多样的非专家轨迹时，常出现两类问题：忽略指令动作，或生成视觉上失真的滚动结果。

进一步地，本文提出 **WorldSync**，从动作分布覆盖、表示层动力学 grounding 和干预效果对齐三个方面增强动作跟随能力：扩展动作后果的训练分布；通过动作强制专家将中间视频表示与机器人动力学联系起来；并约束动作干预引起的预测变化与真实未来变化一致。在 RoboTwin 和真实机器人实验中，WorldSync 提升了 WorldEcho 指标，并作为更可靠的模拟器帮助策略获得更高成功率。

# 2. 方法动机分析

**驱动力**：世界模型若只会复现专家轨迹，就不能可靠模拟策略探索、失败行为或策略更新后的动作，而这些正是策略改进必然产生的状态—动作分布。

**现有痛点**：  
1. 专家示范覆盖的动作后果范围狭窄，模型对可行但未见过的动作缺乏支持；  
2. 视觉逼真不等于动作正确，模型可能生成“看起来合理”但与动作不一致的未来；  
3. 单条轨迹监督只能约束“预测是什么”，不能约束“动作改变时预测应如何改变”；  
4. 仅评估专家动作会掩盖离线/离专家动作下的严重退化。

**核心假设**：动作跟随需要同时解决“见过足够多的动作后果”“视频表示编码了动作诱导的动力学”“不同动作造成的预测差异与真实差异一致”三个问题。

# 3. 方法设计详解

## 3.1 WorldEcho 评测流程

输入初始观测 \(o_0\)、任务指令 \(c\) 和动作序列 \(a_{1:H}\)。模型生成未来视频，模拟器从同一初始状态执行相同动作，得到动作特定的真实视频作为参照。

动作查询分为五类：  
- **示范动作**：当前状态对应的专家动作；  
- **跨状态回放**：从其他状态借用专家动作，检验状态依赖性；  
- **局部扰动**：在专家动作附近加入有限扰动，检验局部敏感性；  
- **策略 rollout**：使用学习策略产生动作，模拟策略偏离专家分布的情况；  
- **可行空间采样**：从更广泛的合法动作空间采样，检验整体可控性。

## 3.2 双重评价

首先进行视觉完整性门控，包括图像质量、运动平滑性、端执行器可见性和机械臂完整性。四项全部通过，才认为视频可用于轨迹评价。

随后用视觉轨迹提取器恢复左右端执行器的位置与旋转，并以加权位置误差和 SO(3) 旋转误差构造局部位姿距离，再通过 NDTW 对齐生成轨迹与真实轨迹，处理时间进度不一致问题。最终对视觉失败样本不直接使用其轨迹误差，而赋予固定惩罚 \(\kappa\)：

\[
S_n =
\begin{cases}
D_{\text{NDTW}}, & \text{视觉通过}\\
\kappa, & \text{视觉失败}
\end{cases}
\]

该设计将“视频不可用”和“视频可用但动作错误”统一到一个指标，同时保留原始 NDTW 和视觉通过率用于诊断。

## 3.3 WorldSync 训练机制

基础模型采用 flow matching 生成未来视频潜变量，标准损失为预测噪声方向与真实流方向之间的均方误差。

**(1) 动作覆盖扩展**  
混合专家、跨状态、局部扰动、策略 rollout 和可行空间动作数据，并加入少量目标域真实机器人数据。仿真与真实动作统一表示为机器人基座坐标系下的相对笛卡尔端执行器位姿增量，从而共享动作语义并减小 sim-to-real 差异。

**(2) Action-Forcing Expert（AFE）**  
AFE 从世界模型不同视频块的中间特征中逐步交叉注意力读取信息，并预测未来端执行器 SE(3) 轨迹。其损失为预测轨迹与真实轨迹的位姿表示误差。AFE 不直接读取动作，也不参与推理；它通过辅助梯度迫使视频特征编码“动作造成的机器人运动”。

**(3) Intervention-Effect（IE）监督**  
对同一观测、同一指令下的两条不同动作分支使用相同噪声，隔离动作条件的影响。模型预测差异与真实未来潜变量差异分别为：

\[
\Delta_\theta=v_\theta^A-v_\theta^B,\quad
\Delta^*=x_0^B-x_0^A
\]

并最小化 \(\|\Delta_\theta-\Delta^*\|_2^2\)。它不再只要求每条结果单独正确，而是要求“动作变化导致的预测变化”也正确。

联合目标为：

\[
\mathcal L=\mathcal L_{FM}
+\lambda_{AFE}\mathcal L_{AFE}
+\lambda_{IE}\mathcal L_{IE}.
\]

# 4. 方法对比与适用性

其本质区别不是单纯提升视频质量，而是把动作跟随定义为**视觉有效性 + 物理轨迹一致性 + 反事实变化一致性**。相较于只回放专家轨迹或只使用视觉相似度的方法，WorldEcho 提供动作特定的 SE(3) 真实参照；WorldSync 则分别从数据、表示和关系监督修复问题。

最重要的创新是 IE：它直接监督动作干预造成的变化，理论上比独立拟合每条轨迹更能抑制“忽略动作、依赖状态先验”的问题。该方法适合机器人策略评估、world-model rollout、VLA 离线/在线改进以及有模拟器可生成动作特定真实轨迹的任务。

# 5. 实验分析

作者在 50 个 RoboTwin 任务、真实堆杯任务及策略迭代实验中验证方法。代表性结论：  
1. 专家动作上的评估会系统性低估模型误差；离专家动作会同时增加轨迹误差和视觉失败。  
2. WorldSync 在完整门控误差和视觉通过率上取得最佳或近最佳表现，并使仿真策略成功率由约 52% 提升至 65%，真实机器人由 48% 提升至 68%。

**优势**：诊断维度完整；动作查询覆盖广；IE 能直接强化动作依赖。  
**局限**：依赖高质量真实轨迹、端执行器提取器和视觉判定器；仍主要在 RoboTwin 和短时操作任务上验证；固定惩罚 \(\kappa\) 与阈值可能影响排名；AFE 增益依赖轨迹标签，且真实开放环境中的长时误差累积尚未解决。

# 6. 实用指南

文中未说明官方代码或数据已开源。复现需准备：多类别动作查询、同初态模拟回放、视频生成 backbone、端执行器轨迹标注/提取器、视觉完整性判定器，以及共享 SE(3) 动作表示。实现时应固定视觉阈值和 \(\kappa\)，区分宏平均与样本平均，并对 IE 配对样本使用相同噪声。迁移到其他任务时，需替换机器人动作参数化、轨迹提取器和可行性过滤器；若没有可执行模拟器，可用真实干预数据或动力学模型构造配对监督。

# 7. 总结

**核心思想：用动作干预约束世界模型真正听懂动作。**

**速记版 pipeline**：  
1. 从专家到随机合法动作构造多种查询；  
2. 对同一动作生成模型视频和真实回放；  
3. 同时检查视频是否有效、机械臂轨迹是否一致；  
4. 用扩展数据、轨迹辅助监督和动作差分监督训练；  
5. 将更可靠的模型用于策略迭代改进。

**Key Findings:**

- To address this gap, we introduce WorldEcho, which probes action following over a broader action distribution using visual integrity and SE(3) trajectory alignment.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24885v1)
- [arXiv](https://arxiv.org/abs/2608.24885v1)

---

<a id='2608.24882v1'></a>
## [Latent Action as Intention Enables Efficient Future Imagination for World Action Models](https://arxiv.org/abs/2608.24882v1)

**Authors:** Xiang Li, Yupeng Zheng, Songen Gu, Huailiang Ma, Feng Yu, Xian Nie, Shanshuai Yuan, Yujie Zang, Weize Li, Shuai Tian, Moyang Liu, Ya-Qin Zhang, Wenchao Ding

**Published:** 2026-08-25

**Categories:** cs.RO

**Abstract:**

World action models (WAMs) improve robot control by modeling how observations evolve, but generating future observations at test time incurs substantial latency. Fast-WAM removes this process for efficiency; however, our matched implementations show lower generalization for Fast-WAM than for future-aware alternatives, especially with scarce robot demonstrations and in out-of-distribution scenarios. To bridge this gap, we introduce **LAWA**, a WAM architecture that uses compact latent actions as an operational representation of future intentions, enabling efficient test-time future imagination without generating future observations. Specifically, a discrete tokenizer enhanced by action-free pre-training produces manipulation-centric codebook targets. LAWA jointly denoises a continuous latent state anchored to these targets with executable action chunks while omitting the future-video branch at inference. On RoboCasa, LAWA achieves state-of-the-art average success rates of 65.6% and 80.8% in the few-shot and full data settings, improving over the matched Fast-WAM baseline by 9.6 and 4.5 points, respectively. It also preserves the performance level of the matched Joint-WAM variant while requiring 42.9% lower inference latency. LAWA also demonstrates competitive zero-shot robustness on LIBERO-Plus and superior performance on real-world tasks. These results show that future imagination need not be discarded: retaining it with compact latent actions yields an effective trade-off among performance, generalization, and latency. Code and models will be released.

**Analysis:**

## 1. 摘要翻译

世界动作模型（WAM）通过建模观测如何演化来提升机器人控制，但测试时生成未来观测会带来较高延迟。Fast-WAM虽在推理阶段移除了未来视频分支，却表现出较弱的泛化能力，尤其是在少量示范和分布外场景中。本文提出LAWA：将紧凑的潜在动作作为未来意图，在不生成未来观测的情况下实现高效未来想象。其核心包括：利用动作无关预训练和掩码监督学习离散潜在动作分词器；在策略训练中，使模型联合去噪连续潜在状态与可执行动作块；推理时丢弃未来视频分支，仅联合预测潜在意图和动作。LAWA在RoboCasa少样本和全数据设置下分别达到65.6%和80.8%的成功率，相比Fast-WAM提升9.6和4.5个百分点；同时接近Joint-WAM性能，但延迟降低42.9%。实验还表明，该方法在LIBERO-Plus和真实机器人任务中具有更好的鲁棒性。

## 2. 方法动机分析

**驱动力：**未来建模有助于理解任务进展，但视频生成太慢；完全移除未来想象又损害少样本泛化。作者的核心假设是：控制所需的未来信息主要是“接下来要发生什么”，不必重建完整视觉细节，因此可用紧凑潜在动作序列表示未来意图。

**痛点：**Joint-WAM推理时迭代生成未来视频，计算昂贵；Fast-WAM只保留训练期视频协同学习，测试时缺少显式未来路径；普通视频重建还容易关注静态外观，忽略手、机械臂和接触区域等细粒度动态。

## 3. 方法设计详解

### （1）潜在动作分词器

输入连续观测 \(o_{t:t+H}\)。首先用DINOv2提取各帧patch特征，再通过非因果时空Transformer获得每帧特征 \(F_k\)。用相邻帧差分 \(F_k-F_{k-1}\) 表示动态变化，并压缩为 \(L\) 个潜在token \(h_k\)。每个token量化为最近的码本向量：

\[
j^*=\arg\min_j\|h-e_j\|^2,\quad l=e_{j^*}.
\]

因此，潜在动作不是机器人关节动作，而是“视觉状态如何变化”的离散编码。前向解码器将其与上一帧视觉token结合，重建下一帧，迫使码本保留转移动力学，而不是只编码静态外观。

### （2）操控中心的监督与动作无关预训练

作者使用SAM 2自动生成手或机械臂掩码。掩码解码器以视觉重建token和潜在动作作为输入，预测交互区域，联合重建损失、感知损失及BCE/Dice/IoU掩码损失训练分词器。这样无需人工标注，却把表示偏向操控相关区域。

分词器同时在机器人视频和第一视角视频上预训练。为避免不同数据源动作速度差异造成训练冲突，作者进行按数据源的帧采样和加权重采样，并保持机器人视频约20%的采样比例。预训练完成后冻结分词器。

### （3）LAWA策略模型

训练时输入当前观测、语言指令及未来观测，建立三个分支：

- 视频分支：预测未来视频latent；
- 潜在动作分支：预测分词器产生的未来潜在动作；
- 动作分支：预测真实可执行动作块。

三者通过联合注意力交互，但使用结构化掩码限制信息流：当前观测不能看到未来；潜在动作只能访问当前观测和潜在序列；动作token可访问当前观测、潜在意图和动作序列。由此避免未来视频泄漏，同时允许动作专家利用逐步形成的未来意图。

三个分支分别加入高斯噪声，并采用flow matching学习从噪声回到目标。总损失为：

\[
L=\lambda_{\rm vid}L_{\rm vid}+
\lambda_{\rm lat}L_{\rm lat}+
\lambda_{\rm act}L_{\rm act}.
\]

注意，码本目标是离散的，但推理时潜在分支在码本嵌入空间中连续去噪，不做最近邻投影；它是离散目标的连续松弛。

**推理阶段：**仅编码当前观测并缓存其特征，删除未来视频分支，从噪声联合去噪潜在意图和动作块，最终输出动作。相比生成图像，这保留了未来规划信号，却显著降低计算量。

## 4. 方法对比与创新

LAWA与Fast-WAM的根本区别在于：Fast-WAM测试时没有未来表示，LAWA保留了在线未来想象；与Joint-WAM相比，LAWA想象的是任务相关潜在转移，而非完整未来画面。主要创新是“潜在动作=可执行控制的未来意图接口”、掩码监督的无动作分词器，以及潜在意图与动作的联合去噪机制。

它最适合少样本、长时序、视觉扰动明显且要求低延迟的机器人操控。局限在于性能依赖分词器质量和预训练视频分布；潜在动作可能丢失精确几何信息，且训练阶段仍需视频分支，整体系统和调参复杂度不低。

## 5. 实验分析

作者在RoboCasa、LIBERO-Plus和xArm7真实任务上，与Fast-WAM、Joint-WAM及多种VLA比较。代表性结论是：RoboCasa上LAWA达到65.6%/80.8%，明显优于Fast-WAM并接近或超过Joint-WAM；其推理延迟为338.5 ms，较Joint-WAM降低42.9%。但去除自中心视频预训练后，LAWA性能低于Joint-WAM，说明潜在表示并非天然优于视觉预测。

## 6. 实用指南

论文声明代码和模型将发布，但当前文本未提供可直接复现的完整配置。复现关键是：先用机器人与第一视角无动作视频训练DINOv2差分量化分词器；加入前向重建和SAM 2掩码损失；冻结分词器后训练三分支flow matching，并严格实现注意力掩码。迁移到其他任务时，应重新收集覆盖目标动态的无动作视频，调整码本规模、潜在序列长度和掩码类别，并保留当前观测—未来意图—动作的接口。

## 7. 总结

**核心思想：**用潜在动作表达未来意图。

**速记版Pipeline：**

1. 从连续视频中提取相邻帧变化。  
2. 将变化压缩并量化成潜在动作码。  
3. 用前向重建和交互区域掩码塑造码的含义。  
4. 联合预测未来意图与真实动作。  
5. 推理时不生成视频，只根据意图生成动作。

**Key Findings:**

- To bridge this gap, we introduce **LAWA**, a WAM architecture that uses compact latent actions as an operational representation of future intentions, enabling efficient test-time future imagination without generating future observations.
- On RoboCasa, LAWA achieves state-of-the-art average success rates of 65.6% and 80.8% in the few-shot and full data settings, improving over the matched Fast-WAM baseline by 9.6 and 4.5 points, respectively.
- These results show that future imagination need not be discarded: retaining it with compact latent actions yields an effective trade-off among performance, generalization, and latency.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24882v1)
- [arXiv](https://arxiv.org/abs/2608.24882v1)

---

<a id='2608.24680v1'></a>
## [Game2World Engine: Unlocking In-the-Wild Gameplay Videos for World Model Training](https://arxiv.org/abs/2608.24680v1)

**Authors:** Wenxuan Shen, Dongna Jin, Dongping Chen

**Published:** 2026-08-25

**Categories:** cs.CV

**Abstract:**

Video games provide a scalable source of training data for video world models, offering diverse environments, complex interactions, and abundant in-the-wild gameplay videos. However, raw gameplay footage entangles the game world with screen-space interfaces, introducing game-specific biases and irrelevant dynamics that hinder world-model training. To address this problem, we introduce GameUI-Taxonomy and G2WEngine, a full-stack framework that formalizes gameplay UI grounding and removal. G2WEngine automatically extracts reusable UI assets from real gameplay videos and synthesizes temporally coherent UI overlays on clean footage. Using this engine, we construct Game2World, comprising 96K synthetic paired videos with precise reconstruction targets and 1,079 in-the-wild clips from 303 games for realistic evaluation. Its asset library contains 5,132 verified UI elements across 21 taxonomy categories, collected from 1,010 representative gameplay frames. Based on Game2World, we propose GameCleaner, a mask-free gameplay UI removal model that combines multimodal semantic understanding with video editing capabilities. Unlike mask-based methods, GameCleaner directly identifies and removes diverse HUD elements while preserving the underlying scene content and temporal dynamics. In a controlled pilot, world models trained on UI-free gameplay improve overall VideoReward by 6.83% over those trained on UI-overlaid data. On UI-removal evaluation, GameCleaner achieves an average AAR of 95.36 on synthetic videos, outperforming the strongest temporal mask baseline by 57.3%, and obtains the best in-the-wild AAR of 80.05 with 99.8 background preservation. These results demonstrate the scalable potential of transforming Internet gameplay videos into high-quality world-model training data. Code, dataset, and model will be available at https://github.com/Dongping-Chen/Game2World.

**Analysis:**

## 1. 摘要翻译

电子游戏为视频世界模型提供了可扩展的数据来源，具有丰富环境、复杂交互和大量真实游戏视频。然而，原始游戏画面将游戏世界与屏幕空间界面混合在一起，引入了游戏特有偏差和无关动态，阻碍世界模型训练。为此，作者提出 **GameUI-Taxonomy** 与 **G2WEngine**，用于统一建模、定位和移除游戏界面。G2WEngine 从真实视频中提取可复用UI资产，并在干净视频上合成时间一致的UI覆盖层，构建包含96K对合成视频和1,079个真实视频片段的Game2World数据集。基于此，作者进一步提出无需掩码的 **GameCleaner**，结合多模态语义理解和视频编辑能力，直接识别并移除多样HUD，同时保持场景内容与时间动态。实验表明，使用无UI视频训练世界模型可使VideoReward提升6.83%；GameCleaner在合成数据上的AAR达到95.36，在真实数据上达到80.05，并保持99.8%的背景内容。

## 2. 方法动机分析

**驱动力**：游戏视频虽规模大、动态丰富，但HUD、菜单、弹窗、水印等并非世界状态，却会成为模型学习的“捷径”。作者的核心假设是：**先进行界面—世界解耦，再训练世界模型，能获得更纯粹的环境动态监督**。

**现有痛点**：传统视频修复主要处理自然视频中的物体，依赖人工掩码，难以判断哪些元素属于UI；游戏UI具有固定屏幕位置、类别语义和独特时间行为，且不同游戏布局差异巨大。简单删除还可能产生明显伪影或破坏背景。

## 3. 方法设计详解

### （1）G2WEngine数据引擎

1. **定义Taxonomy**：将UI划分为21类，包括地图雷达、罗盘、玩家状态、弹药、库存、准星、任务、通知、字幕、聊天、水印和直播叠加等，使标注、采样和评测具有统一语义。
2. **资产提取**：从303款游戏、1,010个代表性关键帧中采样；GPT-5.6 Terra预测UI框和类别，人工校验后，将UI区域抠取为透明资产，并保存类别、来源、位置和渲染元数据。
3. **清洁视频筛选**：将视频切成不重叠的5秒片段，用CLIP特征进行球面K-means聚类，采用簇均衡采样避免某些游戏占据数据集；再以余弦相似度阈值0.94去除相邻冗余片段，统一为720p、30FPS。
4. **UI合成**：从清洁视频出发，按类别概率采样持久HUD与瞬时弹窗。使用类别空间先验和三种布局模板确定位置；加入确定性位置抖动、碰撞检测、透明度控制及运动轨迹。弹窗还具有淡入、滑入、擦除、弹出和边界弧线等动画，从而生成时间连续而非逐帧随机的覆盖层。
5. **完整监督输出**：每个样本同时保存清洁视频、带UI视频、逐帧二值掩码、实例框、类别、轨迹和渲染记录，因此可用于检测、分割、跟踪和移除。

### （2）GameCleaner结构

输入为带HUD的视频和文本指令，可选输入清洁参考图。模型由冻结的预训练多模态大模型（MLLM）和视频扩散Transformer（DiT）组成。MLLM联合理解视频帧与指令，通过可学习查询提取“哪些区域是UI”的语义信息，再经连接器注入DiT。与此同时，源视频VAE特征通过时间相关残差加入噪声目标，帮助维持原始人物、几何结构和运动。

流匹配损失为  
\[
\|v_\theta(z_t,t,c)-(z_1-z_0)\|_2^2
\]
即让模型学习从噪声状态逐步生成清洁目标视频。直观上，MLLM负责“识别并决定删什么”，DiT负责“补回被遮挡内容并保持时序”，源视频残差负责“尽量不要改动无关场景”。

## 4. 对比与创新

其本质区别不是单纯的视频修复，而是把HUD移除定义为**接口—世界解耦问题**。创新包括：统一的游戏UI语义体系；从真实游戏中提取资产并进行可控、时间一致的合成；无需人工掩码的跨游戏移除模型；同时评价移除成功、伪影和背景保持，而非只统计“UI是否消失”。

适合互联网游戏视频清洗、世界模型预训练、游戏视频编辑及HUD理解任务；对大型不透明面板、快速变化UI和与场景强耦合的界面较不理想。

## 5. 实验分析

作者以5,442段相同视频构造有UI/无UI训练集，结果显示无UI训练使世界模型整体VideoReward提升6.83%。GameCleaner在合成集AAR达95.36，在真实集参考图模式下AAR为80.05、背景保持99.80%，说明其优势在于兼顾删除质量与场景保真，而非激进擦除。

主要局限是依赖合成监督，真实世界缺少清洁参考；当前下游验证仍是文本条件视频生成，而不是完整动作条件世界模型。真实视频中的字幕、HUD与场景纹理混叠也可能导致误判。

## 6. 实用指南

论文声明代码、数据和模型将发布于`Dongping-Chen/Game2World`。复现关键是：收集无UI游戏片段；按Taxonomy标注并人工校验资产；使用5秒、720p、30FPS格式合成；训练时冻结MLLM与连接器，仅对DiT使用rank=64的LoRA，学习率1e-4、全局batch 32。评测需按时间戳对齐，统一缩放到1280×720，并同时报告UI移除、AAR、伪影和背景保持。该框架可迁移到直播水印、字幕、移动端游戏控件等屏幕叠加移除任务，只需扩展类别体系和空间/时间先验。

## 7. 总结

**核心思想：先解耦游戏界面，再学习世界动态。**

**速记版Pipeline**：  
1. 统一定义游戏UI类别；  
2. 从真实视频抠取可复用界面素材；  
3. 在干净游戏画面上合成带时间变化的HUD；  
4. 训练模型识别并清除HUD，同时补全背景；  
5. 用清除质量和场景保真共同评估。

**Key Findings:**

- To address this problem, we introduce GameUI-Taxonomy and G2WEngine, a full-stack framework that formalizes gameplay UI grounding and removal.
- Based on Game2World, we propose GameCleaner, a mask-free gameplay UI removal model that combines multimodal semantic understanding with video editing capabilities.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24680v1)
- [arXiv](https://arxiv.org/abs/2608.24680v1)

---

<a id='2608.24618v1'></a>
## [VIP: Variation-based Iterative-learning Planning for Robotic Navigation](https://arxiv.org/abs/2608.24618v1)

**Authors:** Shuli Lv, Pengda Mao, Chen Min, Li Hong, Runxiao Liu, Shuai Wang, Quan Quan

**Published:** 2026-08-25

**Categories:** cs.RO

**Abstract:**

Over the past decade, autonomous robotic systems have been increasingly deployed in applications such as surveying, search and rescue, and last-mile delivery. These applications require robots to generate safe and efficient motion plans in large, complex, and obstacle-dense environments, often under limited onboard computing resources. However, conventional planning methods commonly rely on finite-dimensional trajectory parameterization or increasingly long prediction horizons, leading to rapidly growing computational costs, particularly in multi-robot scenarios. This paper presents a novel variation-based iterative-learning planning (VIP) framework for efficient motion planning of both single robots and robotic swarms. Instead of optimizing a large number of discrete trajectory variables, VIP directly updates the planning command as a continuous function in an infinite-dimensional function space. The same variation-based update can be implemented in a model-in-the-loop manner for offline planning or in a robot-in-the-loop manner between online physical executions. By avoiding the computational burden associated with horizon expansion and high-dimensional trajectory discretization, VIP maintains a per-iteration computational complexity of $\mathcal{O}(n)$, where $n$ denotes the number of spatial discretization points. Extensive simulations and real-world experiments demonstrate that the proposed framework can efficiently generate and iteratively improve motion plans for different planning objectives, robotic platforms, and swarm configurations, highlighting its effectiveness, computational efficiency, and scalability as a general planning methodology.

**Analysis:**

# 1. 摘要翻译

过去十年，自主机器人已广泛应用于测绘、搜救和末端配送等任务。这些任务要求机器人在大规模、复杂且障碍密集的环境中生成安全高效的运动规划，同时受到机载计算资源有限的约束。传统方法通常依赖有限维轨迹参数化或不断扩大的预测时域，导致计算成本迅速增加，尤其是在多机器人场景下。本文提出一种面向单机器人和机器人群体的**基于变分的迭代学习规划（VIP）框架**。VIP不再优化大量离散轨迹变量，而是在无限维函数空间中直接更新规划指令这一连续函数。同一变分更新既可通过模型在环仿真实现离线规划，也可在多次真实执行之间通过机器人在环实现在线学习。借助避免时域扩展和高维轨迹离散化，VIP每次迭代的计算复杂度为 \(O(n)\)，其中 \(n\) 为空间离散点数。仿真和真实实验表明，VIP能够针对不同目标、机器人平台及群体规模高效生成并持续改进运动规划。

# 2. 方法动机分析

**驱动力**：作者认为规划的主要瓶颈不是某个优化器不够快，而是“把整条轨迹离散成大量变量”这一问题表述本身。

**现有痛点**：MPC、MPCC和多项式规划需联合优化状态、控制量或轨迹系数，并处理动力学、连续性和避障约束；时域或轨迹分辨率增加后，计算和耦合约束迅速增长。采样/反应式方法虽快，却通常缺乏全局效率优化。多机器人规划还会引入联合状态空间、通信和碰撞协调开销。

**核心假设**：机器人已经拥有可行的几何路径或虚拟管道；路径跟踪误差、群体密度误差都可压缩为一个标量“能量”；机器人前进速度越激进，越可能牺牲误差调节能力。因此，只需学习“沿路径何处快、何处慢”，而无需重新优化完整轨迹。

# 3. 方法设计详解

## 3.1 总体Pipeline

1. **空间可行性生成**：用Tube-RRT*等方法生成无碰撞路径 \(\gamma(l)\) 或虚拟管道 \(\mathcal T\)，其中 \(l\) 是路径弧长坐标。VIP不负责决定“往哪里走”，而负责决定“以多快速度走”。

2. **构造调节能量**：  
   - 单机器人：以路径最近点误差 \(e_p\) 定义 \(V_s=\frac12\|e_p\|^2\)。  
   - 机器人群体：用KDE由机器人位置估计密度 \(\rho\)，与期望密度 \(\rho_d\) 比较，定义 \(V_m=\frac12\int(\rho-\rho_d)^2dp\)。

3. **组合运动指令**：将指令分成两部分：
   \[
   v_c'=v_n'+v_t'
   \]
   其中 \(v_n'\) 用于纠正路径/密度误差，\(v_t'=v_t(l)t_c\) 用于沿路径前进。随后通过速度饱和得到可执行指令 \(v_c=\mathrm{sat}(v_c',v_m)\)。这一步体现了VIP的关键权衡：提高 \(v_t\) 可能减少误差调节所剩的饱和裕度。

4. **时间域转空间域**：利用
   \[
   D_t=vD_l,\qquad D_lV=\dot V/v
   \]
   把随时间变化的系统改写为沿路径位置变化的能量动力学。统一形式为
   \[
   \dot V=-\lambda_1\kappa_Ev_n^p+\epsilon v_t^ov_n^q+\epsilon_1,
   \]
   即主调节项降低能量，前进速度会产生耦合影响。

5. **模型型变分优化**：定义代价
   \[
   J=\int_0^L\left(\frac1{\kappa_Ev_t}+k_VV\right)dl。
   \]
   第一项代表通行时间，第二项惩罚路径/群体偏差。通过Hamiltonian和伴随变量计算泛函梯度 \(F_n(l)\)，并按
   \[
   v_{t,k+1}=v_{t,k}-b_1F_{n,k}
   \]
   更新速度函数。它是理论上的精确下降方向。

6. **模型无关迭代学习**：真实实现不计算动力学、Jacobian或伴随方程，而记录第 \(k\) 次执行中的能量曲线 \(V_k(l)\)，令 \(\zeta_k=k_eV_k\)，使用分段函数 \(g(\zeta)\)：
   - 能量低于容许区间：\(g<0\)，提高速度；
   - 能量高于区间：\(g>0\)，降低速度；
   - 位于区间内：\(g=0\)，保持速度。  
   更新为
   \[
   v_{t,k+1}(l)=v_{t,k}(l)-b_3g(k_eV_k(l)).
   \]

7. **重复执行与收敛**：模型在环模式用仿真回放，机器人在环模式用真实飞行记录。不同执行时长通过空间坐标 \(l\) 对齐，逐点更新速度剖面。

## 3.2 模块协同

几何规划负责安全可行区域；误差调节场负责局部稳定与分布保持；VIP只学习共享的前进速度。对群体而言，KDE和扩散场实现宏观密度调节，虚拟管道控制器实现个体避碰和边界约束，形成“宏观加速、微观维稳”的分工。

# 4. 方法对比与创新

本质区别在于：主流方法优化有限维轨迹变量，VIP直接优化空间上的速度函数；主流方法依赖预测模型和在线数值求解，VIP可从执行结果中无模型更新；群体场景中VIP学习一个公共速度剖面，而非为每个机器人分别规划轨迹。

主要创新包括：统一单机器人误差与群体密度误差的能量表述；将迭代学习从“轨迹跟踪层”提升到“规划指令层”；以能量记录近似变分方向，并实现 \(O(n)\) 的逐点更新。

最适合重复经过相同或相似路径、执行代价可接受、已有安全路径/管道的无人机和群体导航。它不适合完全陌生的拓扑决策、强动态障碍或一次性任务。

# 5. 实验分析

作者通过单机未知环境仿真、二维/三维群体仿真，以及LiDAR无人机、升力翼四旋翼和三机群真实实验验证。代表性结果是：VIP在单机比较中将重规划时间降至约2.37–2.62 ms，并较MPCC和多项式方法显著降低计算开销；三机群真实飞行中，遍历时间由241 s降至82 s。

优势是计算轻量、能利用真实执行误差、能迁移到不同平台和群体。局限是安全性仍主要依赖外部路径/虚拟管道控制器；“\(O(n)\)”只描述更新开销，不包括KDE、感知、路径规划和整次rollout；模型无关更新要求重复执行同一路径，且能量区间和步长依赖经验调参。

# 6. 实用指南

论文提供代码：`github.com/lyushuli/VIP`。复现需先生成路径/虚拟管道，再计算空间误差能量，记录 \(V_k(l)\)，设计 \(g\) 的能量死区、\(k_e\) 和学习率 \(b_3\)，并对速度施加上下限及平滑约束。群体实现还需选择KDE带宽、密度正则项和期望密度前移量。迁移到其他任务时，只需替换能量定义和误差调节场，保留“空间记录—能量反馈—速度剖面更新”机制。

# 7. 总结

**核心思想：用执行误差学习路径速度。**

**速记版Pipeline**：  
1. 先生成安全可行路径；  
2. 沿路径记录跟踪或群体分布误差；  
3. 误差小的位置加速，误差大的位置减速；  
4. 多次执行后逐点更新速度曲线；  
5. 由底层控制器负责最终安全与稳定。

**Key Findings:**

- This paper presents a novel variation-based iterative-learning planning (VIP) framework for efficient motion planning of both single robots and robotic swarms.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24618v1)
- [arXiv](https://arxiv.org/abs/2608.24618v1)

---

<a id='2608.24603v1'></a>
## [Gripper-aware Vision Language Action Models](https://arxiv.org/abs/2608.24603v1)

**Authors:** Hanyi Zhang, Zihong Luo, Tianyu Li, Khang Nguyen, Basu Hela, Shreyas Kumar, Ngoc Duy Tran, Feng Dai, Charith Munasinghe, Jorge Peña Queralta, Giovanni Toffetti, Khoa Vo, Ngan Le, Ravi Prakash, Quan Vuong, Tung D. Ta, Long Hu, Anh Nguyen, Baoru Huang

**Published:** 2026-08-25

**Categories:** cs.RO

**Abstract:**

Vision language action models (VLAs) have advanced general purpose robotic grasping and manipulation by enabling robots to interpret visual observations and natural language instructions to generate executable action sequences. However, existing VLAs often implicitly assume gripper invariance, despite grasping strategies being inherently embodiment-dependent. Different gripper types, such as parallel-jaw and suction, usually require distinct interaction strategies to achieve the same grasping objective. Moreover, current datasets for VLAs predominantly rely on parallel-jaw grippers, limiting gripper-aware learning. To address this gap, we introduce MiGA, a multi-gripper-aware dataset spanning five distinct gripper types across multiple robots with 103,000 demonstrations, explicitly capturing strategy divergence under shared task objectives. We further propose GVLA, which combines a new multi-gripper tokenizer with adapter-based policy routing. Our new gripper encoding induces structured embedding information that balances parameter sharing and strategy differentiation, while layer-wise probing confirms meaningful gripper-conditioned representations for VLAs. Intensive experiments in both simulation and real-world robots show that our GVLA outperforms the current baselines across evaluated settings. Our method also improves zero-shot generalization or few-shot adaptation to new objects or unseen tasks, and enable more efficient gripper adaptation.

**Analysis:**

### 1. 论文主要贡献概述

本文针对视觉-语言-动作模型（VLA）通常忽略末端执行器差异的问题，提出了多夹爪感知数据集 **MiGA** 和模型 **GVLA**。MiGA 包含五类夹爪、多个机器人平台和约 103,000 条操作示范，GVLA 则通过多夹爪 tokenizer 与基于适配器的策略路由，使模型能够在共享任务目标的同时学习不同夹爪对应的操作策略。

### 2. 关键创新与方法

#### （1）构建多夹爪、多机器人数据集 MiGA

现有 VLA 数据集大多使用平行夹爪，导致模型容易将“完成抓取任务”与某一种固定的运动策略绑定。MiGA 的核心价值在于：

- 覆盖五种不同夹爪类型；
- 包含多个机器人平台；
- 收集约 103,000 条示范数据；
- 在相同任务目标下，显式记录不同夹爪产生的策略差异。

例如，平行夹爪可能需要对物体进行夹持，而吸盘夹爪可能需要选择更平坦、可吸附的表面。这种数据设计能够让模型学习“任务目标相同，但动作策略依赖具身形态”的关系。

#### （2）提出多夹爪 tokenizer

GVLA 使用新的夹爪编码机制，将夹爪类型信息注入 VLA 的输入或中间表示中。其目标不是简单地把不同夹爪视为完全独立的模型，而是使表示空间同时具备：

- **参数共享**：不同夹爪之间共享通用的视觉、语言和任务知识；
- **策略区分**：在实际动作生成阶段，根据夹爪能力和接触方式产生不同策略。

这类似于在模型内部引入一种显式的 embodiment conditioning，使动作预测不仅依赖图像和语言，也依赖当前可用的末端执行器。

#### （3）基于适配器的策略路由

论文进一步采用 adapter-based policy routing，根据夹爪条件将动作生成过程路由到相应的策略模块。相比为每种夹爪训练完全独立的模型，这种方式可能具有更好的参数效率，并支持：

- 不同夹爪之间的知识迁移；
- 新夹爪的快速适配；
- 在共享任务知识基础上的策略专门化。

#### （4）分层表示探测与实验验证

作者使用 layer-wise probing 检查模型不同层是否学习到了有意义的夹爪条件表示。这一点比较重要，因为它不仅展示最终任务成功率，还试图验证模型内部是否真正编码了夹爪相关信息，而不是仅依赖数据分布中的偶然相关性。

### 3. 对领域的潜在影响

#### 对机器人学习的影响

该工作将“末端执行器类型”从一个隐含的实验配置因素，提升为 VLA 中的显式条件变量。这有助于推动机器人策略从“任务级泛化”发展到“任务—机器人形态联合泛化”。

其潜在影响包括：

- 提高同一模型在不同机器人和夹爪上的可迁移性；
- 降低更换夹爪后重新采集数据和训练模型的成本；
- 支持针对新夹爪的零样本或少样本适配；
- 促进通用机器人基础模型向多具身、多平台方向发展。

#### 对计算机视觉的影响

从计算机视觉角度看，这项工作不仅关注“看到了什么物体”，还关注“当前机器人能够以什么方式与物体交互”。因此，它将视觉表示从静态目标识别扩展到了：

- 物体的可抓取区域；
- 与夹爪接触相关的几何属性；
- 物体表面是否适合吸附或夹持；
- 视觉观察与动作策略之间的具身依赖关系。

这对视觉表示学习、视觉伺服、可供性预测（affordance prediction）以及视觉-语言-动作对齐都具有潜在意义。尤其是，同一个物体在不同夹爪下可能对应不同的抓取点、接近方向和运动轨迹，模型需要学习这种条件化的视觉可供性。

### 4. 可能受益的相关领域与应用

#### 相关研究方向

- 视觉-语言-动作模型（VLA）
- 机器人基础模型和具身智能
- 多机器人策略学习
- 机器人操作与抓取规划
- 视觉可供性学习
- 视觉伺服与接触感知
- 模仿学习和离线强化学习
- 多任务和跨具身迁移学习
- 参数高效微调与模块化神经网络
- sim-to-real 机器人学习

#### 潜在应用场景

- **仓储与物流**：根据物体形状、材质和表面选择平行夹爪、吸盘或柔性夹爪；
- **制造业**：在不同工业机械臂和末端工具之间复用操作策略；
- **电商分拣**：处理形状、尺寸和材质差异较大的物品；
- **农业机器人**：针对易损果实、枝叶或不规则物体选择合适的抓取方式；
- **家庭服务机器人**：在不确定环境中利用不同工具完成开门、拾取和整理；
- **医疗和实验室自动化**：针对脆弱、微小或污染敏感对象采用不同接触策略；
- **机器人平台快速部署**：新安装一种夹爪后，通过少量示范快速适配。

### 5. 根据摘要可以推断的局限性

以下局限性并不一定代表论文实验未覆盖，而是根据摘要尚无法确认、并值得进一步考察的方面。

#### （1）夹爪覆盖范围仍然有限

MiGA 覆盖五种夹爪，但真实机器人末端执行器种类非常多，例如柔性夹爪、多指手、磁吸工具、工具更换系统和复杂组合式执行器。因此，模型是否能够推广到训练中未出现的夹爪形态仍不明确。

#### （2）夹爪类型并非唯一的具身因素

动作策略还会受到以下因素影响：

- 机械臂自由度和运动学结构；
- 负载能力；
- 夹爪尺寸和开合范围；
- 触觉传感器配置；
- 控制频率和动作空间定义；
- 相机位置与视角；
- 机器人末端坐标系和标定误差。

如果模型主要编码的是离散的“夹爪类型”，可能仍不足以处理更连续、更复杂的机器人具身差异。

#### （3）数据集的任务和物体分布可能限制泛化

虽然数据量达到 103,000 条示范，但数据是否覆盖足够丰富的：

- 物体类别；
- 材质、形状和尺寸；
- 遮挡与光照条件；
- 非结构化场景；
- 多步操作和失败恢复情况；

摘要中尚未说明。如果数据主要来自有限任务或相似环境，零样本泛化能力可能会受到分布偏移影响。

#### （4）零样本泛化的适用范围需要进一步澄清

摘要声称模型能够改善对新物体和未见任务的零样本或少样本泛化，但尚不清楚：

- “新任务”与训练任务的差异程度；
- 是否涉及全新的动作组合；
- 是否仅在相同场景或相同机器人上测试；
- 零样本性能相对基线的提升幅度；
- 新夹爪是否真正完全未出现在训练分布中。

这些因素会直接影响结论的强度。

#### （5）视觉信息可能不足以解决所有抓取问题

对于透明、反光、柔软、可变形或表面不均匀的物体，仅凭 RGB 或视觉输入可能难以判断吸附质量、接触稳定性和摩擦条件。若 GVLA 没有结合触觉、力觉或夹爪状态信息，其在复杂接触操作中的鲁棒性可能有限。

#### （6）模块化适配器的扩展性问题

随着夹爪种类和机器人平台增加，策略路由和适配器数量可能不断增长，带来：

- 参数管理复杂度；
- 路由错误；
- 新旧适配器之间的负迁移；
- 多夹爪同时可用时的选择问题。

因此，模型是否能够支持连续变化的执行器属性，可能比单纯增加离散夹爪类别更关键。

#### （7）实验指标与真实部署成本仍需关注

摘要指出在仿真和真实机器人上取得更好结果，但尚未给出成功率、数据效率、推理延迟、训练成本和安全性等具体指标。对于实际机器人部署，动作失败代价、碰撞风险和长时序任务稳定性同样重要。

### 总体评价

这项工作的核心价值在于明确指出：**VLA 的动作策略并不是只由视觉场景和语言目标决定，还受到机器人末端执行器能力的直接约束**。通过多夹爪数据和条件化策略建模，论文将具身差异纳入视觉—语言—动作学习框架，对多机器人泛化、视觉可供性理解和机器人基础模型具有较强的潜在意义。其最终影响将取决于模型能否从有限的夹爪类别推广到更广泛的机器人形态、接触传感器和真实非结构化环境。

**Key Findings:**

- To address this gap, we introduce MiGA, a multi-gripper-aware dataset spanning five distinct gripper types across multiple robots with 103,000 demonstrations, explicitly capturing strategy divergence under shared task objectives.
- We further propose GVLA, which combines a new multi-gripper tokenizer with adapter-based policy routing.
- Our new gripper encoding induces structured embedding information that balances parameter sharing and strategy differentiation, while layer-wise probing confirms meaningful gripper-conditioned representations for VLAs. Intensive experiments in both simulation and real-world robots show that our GVLA outperforms the current baselines across evaluated settings.
- Our method also improves zero-shot generalization or few-shot adaptation to new objects or unseen tasks, and enable more efficient gripper adaptation.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24603v1)
- [arXiv](https://arxiv.org/abs/2608.24603v1)

---

<a id='2608.24485v1'></a>
## [NeuralParker: A Reinforcement Learning Planner for Irregular Parking Environments](https://arxiv.org/abs/2608.24485v1)

**Authors:** Zihan Wang, Bai Huang, Yang Guan, Xiao Li, Haoyu Xu, Naizheng Wang, Shengbo Eben Li

**Published:** 2026-08-25

**Categories:** cs.RO, cs.LG

**Abstract:**

Automated parking commonly assumes marked slots and short approach maneuvers. Delivery and service vehicles, however, may need to reach an operator-specified pose in an irregular bounded environment from a distant start. Existing learning-based parking planners often rely on local observations, which can restrict long-range route reasoning. To address this problem, we present NeuralParker, a reinforcement learning-based hybrid planner for arbitrary-pose parking. NeuralParker encodes full-environment obstacle and boundary geometry in a target-relative vertex representation, allowing the policy to retain route-defining context throughout the approach. It further couples a learned curvature--length arc policy with an in-loop terminal ensemble that selects from diverse cubic Hermite connections using a curvature-regularized cost. We also establish factorial and long-range route-choice benchmarks to evaluate planning success and trajectory quality. Experiments on these benchmarks show that NeuralParker achieves higher planning success and better overall trajectory quality than the evaluated baselines, while ablation studies support the benefits of the target-relative global representation and terminal ensemble. Finally, a real-vehicle evaluation confirms that the planner transfers effectively to real delivery-vehicle perception at a working parking site, planning successfully at low computational cost.

**Analysis:**

# 1. 摘要翻译

自动泊车通常假设存在标准车位，且车辆从附近通道进入。但配送车、服务车往往需要从远距离起点驶入不规则封闭区域中的任意指定位姿。现有学习型泊车规划器多依赖局部观测，难以进行长距离路线推理。为此，本文提出基于强化学习的混合规划器 **NeuralParker**。它以目标为中心，将障碍物和边界的完整几何编码为顶点表示，使策略在接近目标的全过程中保留决定路线的全局信息；同时，策略输出曲率—长度运动弧，并在循环内部通过曲率正则化代价，从多个三次Hermite终端连接中选择可行解。作者构建了阶乘式和长距离路线选择基准。实验表明，该方法在规划成功率和轨迹质量上优于对比方法；消融实验验证了目标相对全局表示和终端候选集的作用；真实车辆实验也表明其能够以较低计算成本迁移到实际配送车场景。

# 2. 方法动机

**驱动力与痛点：**传统Hybrid A*、状态格和优化方法几何约束明确，但每次查询都需搜索或优化；学习方法虽然快速，却常使用局部LiDAR、局部栅格或紧凑状态，目标远处的入口、墙体和绕行结构不可见，容易先选错接近方向。此外，固定终端连接器会限制策略必须到达的“可结束状态”，且终端模块与策略训练脱节。

**核心假设：**若策略始终获得目标侧全局几何，并在训练时直接感知终端连接器的可行性与代价，则它能提前选择正确路线，并以更少倒车和曲率变化完成泊车。

# 3. 方法设计详解

## 3.1 Pipeline

1. **统一坐标变换：**将目标设为原点、目标朝向对齐正 \(y\) 轴，把车辆、障碍物顶点和边界端点转换到目标相对坐标系，因此整体平移和旋转不影响输入语义。  
2. **构造观测：**输入包括车辆位姿、20个障碍物槽位、10个边界槽位及120束360° LiDAR。圆形障碍物用八边形/16段环近似；LiDAR距离减去车体矩形外接 footprint 的方向性偏置，并截断为非负值。几何槽位不足时填充哨兵值，超过容量直接报错。  
3. **场景编码：**车辆、障碍物、边界和LiDAR分别经过MLP；障碍物与边界token用自注意力聚合，再与车辆和LiDAR特征融合，输入Actor/Critic。  
4. **策略推进：**Actor输出曲率 \(\kappa\in[-0.68,0.68]\) 和带符号长度 \(l\in[-10,10]\)。根据恒曲率模型确定弧线轨迹；长度符号决定前进或倒车。每条弧采样30个参考点进行碰撞和边界检查。  
5. **终端集成：**若当前弧无碰撞、未直接到达目标，且目标相对航向满足 \(\sin\tilde\theta>0\)，则生成81条Hermite曲线：9种终端横向/航向扰动，与起点、终点切向尺度 \(\{0.8,1.0,1.2\}\) 的组合。每条曲线采样100点，检查边界、障碍物和最大曲率。  
6. **选择与训练：**在可行曲线中最小化 \(10\Delta\kappa_H+10L_H+5\)，兼顾长度和曲率变化。成功奖励还惩罚倒车次数、曲率突变和终端代价。该终端判断直接进入PPO rollout，而非仅作为事后平滑。训练采用从近到远的地理距离课程。

## 3.2 模块协同与关键新意

本文真正的创新不只是“用Transformer或RL”，而是把**全局几何输入、策略路线选择和终端可行性**放进同一闭环：策略决定长距离前缀，解析连接器负责最后精确到位；连接器的失败会直接影响训练信号，从而反向塑造策略的到达状态。

# 4. 对比与适用性

相较局部LiDAR/BEV方法，NeuralParker显式保留目标侧结构；相较纯Hybrid A*，它用训练阶段摊销重复搜索；相较“先全局规划、再局部控制”的分层方案，它不在固定距离处强行切换，避免交接状态不可终止。最适合静态、二维、地图可获得、目标位姿明确且需要长距离绕行的配送车或服务车泊车。它不是动态交通、强感知不确定性或超大规模地图的直接解决方案。

# 5. 实验分析

作者在阶乘泊车和Topology-Stress两个基准上，与适配后的HOPE及Hybrid A*分层方案比较，并进行全局表示、终端集成和起点课程消融。代表性结果是：阶乘基准成功率约 **97.4%**，优于HOPE约72%；终端候选集使成功率由96.5%提升至97.4%，同时减少路径长度和曲率变化。真实场景中，NeuralParker规划时间仅约0.37 ms，但全车体验证成功率为60%，暴露出训练碰撞模型过于理想化的问题。

# 6. 实用指南

论文未声明公开代码或数据。复现需重点实现：目标相对坐标变换、固定几何槽位、footprint修正LiDAR、恒曲率推进、81候选Hermite连接及100点可行性检查；保持训练和测试使用完全一致的碰撞规则。关键设置包括PPO 240轮、每轮60000步、折扣率0.99、GAE 0.95、曲率上限0.68、200步超时，以及地理距离课程。迁移到机械臂、移动机器人或无人船时，可将顶点改为场景几何token，将恒曲率模型替换为对应动力学，并重新设计终端连接器和可行性代价。

# 7. 总结

**核心思想：全局看路，策略走弧，解析收尾。**

**速记版Pipeline：**
1. 把所有环境信息改写到目标坐标系。  
2. 用注意力网络同时看全局障碍和近场距离。  
3. 输出一段前进或倒车的恒曲率弧线。  
4. 到达合适姿态后尝试81种终端曲线。  
5. 只执行无碰撞且代价最低的完整路线。

**Key Findings:**

- To address this problem, we present NeuralParker, a reinforcement learning-based hybrid planner for arbitrary-pose parking.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24485v1)
- [arXiv](https://arxiv.org/abs/2608.24485v1)

---

<a id='2608.24469v1'></a>
## [Low-Rank Ternary Adaptation for Fine-Tuning Transformers](https://arxiv.org/abs/2608.24469v1)

**Authors:** Alexandru-Dragos Manolache, Yunqiang Li, Jan van Gemert

**Published:** 2026-08-25

**Categories:** cs.CV, cs.LG

**Abstract:**

Ternary transformers offer extreme memory and compute efficiency, but existing low-bit LoRA-based methods cannot directly fine-tune ternary weights. Current approaches either require dequantization, restoring low-bit base weights to higher precision to merge with adaptation weight, or update only quantization parameters, preventing a merged model that remains ternary. We propose ternary multiplicative adaptation, which represents discrete updates of ternary weights such as sign flips or zeroing through a low-rank Kronecker factorization into two small ternary matrices applied element-wise to ternary weights. This design is parameter-efficient and expressive, preserves the ternary domain, and supports direct merging without dequantization. Experiments on six models across language and vision, including ternarized LLaMA-3 1B and 3B and a ternary ViT-B/16, demonstrate that our method recovers much of the performance lost to quantization and outperforms strong low-bit and ternary baselines. Code is available at https://github.com/alexmanoo/ternary_adaptation.

**Analysis:**

## 1. 摘要翻译

三值 Transformer 具有极高的存储与计算效率，但现有基于低比特 LoRA 的方法不能直接微调三值权重。当前方法要么将低比特权重反量化到高精度后再与适配器合并，要么只更新量化参数，因而无法得到合并后仍保持三值的模型。本文提出**三值乘法适配（Ternary Multiplicative Adaptation）**：通过两个较小的三值矩阵的低秩 Kronecker 分解，表示三值权重的离散更新，例如符号翻转或置零，并将其逐元素作用于三值权重。该方法参数高效、表达能力较强，始终保持三值域，并可直接合并而无需反量化。实验覆盖语言和视觉任务，包括三值 LLaMA-3 1B/3B 与三值 ViT-B/16，结果表明该方法能恢复大量量化损失，并优于多种低比特和三值基线。

## 2. 方法动机分析

**驱动力**：三值权重仅取 \(\{-1,0,1\}\)，理论上约为 1.58 bit/weight。若微调后仍维持该域，才能真正保留其存储、计算和部署优势。

**现有痛点**：LoRA采用连续加法
\[
W'=W+\Delta W,
\]
反量化后的权重虽可进行优化，但合并后通常变为浮点数，还需重新量化；而重新量化会引入新的误差。QA-LoRA虽然能避免反量化，却主要修改量化参数，不能保证合并权重仍为严格三值。更根本地说，三值模型的有效更新不是“小幅调整”，而是对非零权重进行**保留、置零或翻转符号**。

**核心假设**：如果将更新改写为三值乘法掩码，并用紧凑结构表达这个掩码，就能在极少参数下实现有效的离散权重重排。

## 3. 方法设计详解

### Pipeline

1. **准备三值骨干**：将预训练矩阵量化为  
   \[
   W_{\text{tern}}\in\{-1,0,1\}^{d_{\text{out}}\times d_{\text{in}}},
   \]
   骨干冻结，量化尺度可单独保存。

2. **构造三值更新掩码**：定义  
   \[
   W'_{\text{tern}}=W_{\text{tern}}\odot\Delta_{\text{tern}}.
   \]
   当 \(\Delta_{ij}=1\) 时保留权重，等于 0 时置零，等于 -1 时翻转符号。因此结果必然仍属于 \(\{-1,0,1\}\)。

3. **Kronecker 参数化**：不直接学习完整掩码，而令  
   \[
   \Delta_{\text{tern}}=A\otimes B,
   \]
   其中 \(A\in\{-1,0,1\}^{p\times q}\)，\(B\in\{-1,0,1\}^{r\times s}\)，并满足
   \[
   pr=d_{\text{out}},\quad qs=d_{\text{in}}.
   \]
   Kronecker积会把 \(A\) 中每个元素替换为其与整个 \(B\) 的乘积，从而用两个小矩阵覆盖完整权重矩阵。参数量为 \(pq+rs\)，方阵情况下可降为约 \(2d\)，而不是 \(d^2\)。

4. **连续代理训练**：实际优化浮点代理 \(\bar A,\bar B\)，前向时通过阈值函数投影：
   \[
   A=\operatorname{Tern}(\bar A),\quad B=\operatorname{Tern}(\bar B).
   \]
   阈值依据矩阵绝对值均值确定，反向传播采用 STE，使离散投影仍可训练。

5. **初始化与合并**：采用 Balanced 或 Normalized 初始化，并通过补偿原始权重符号，使训练初始时模型功能不变。训练结束后计算 \(W_{\text{tern}}\odot(A\otimes B)\)，丢弃代理参数，部署单一三值矩阵。

### 关键设计理解

该方法名义上属于“低秩适配”，但其重点并非传统 LoRA 的低矩阵秩，而是**用少量参数生成一个可能具有高秩的结构化离散掩码**：
\[
\operatorname{rank}(A\otimes B)=\operatorname{rank}(A)\operatorname{rank}(B).
\]
其表达能力来自 Kronecker 结构，离散合法性来自乘法闭包。

## 4. 方法对比与适用场景

与 LoRA/QLoRA 的本质区别是：前者在连续空间做加法，本方法在三值空间做乘法；与 QA-LoRA 的区别是：本方法直接修改权重状态，而不是量化器参数。创新主要包括三点：三值 keep/zero/flip 更新、三值 Kronecker 结构化表示、无需反量化和重新量化的直接合并。

它适合已经量化为三值的 LLM、ViT 等模型，尤其适用于内存受限、要求零推理额外开销、且最终模型必须保持 1.58 bit 的部署场景。

## 5. 实验分析

作者在三值 LLaMA、Falcon、BitNet 和 ViT 上进行语言建模、常识问答、GSM8K 与 ImageNet-100 实验。代表性结论是：在 LLaMA-3.2-3B 上，平均准确率由三值基线的 31.9 提升至 38.3，PPL 由 45.6 降至 22.3；在 ViT 上明显优于重新量化的 QLoRA。

**优势**：参数量极小、合并后零额外推理开销、严格保持三值、避免重新量化误差。  
**局限**：原本为 0 的权重无法重新激活；Kronecker 因子必须整除层维度，结构可能限制更新模式；训练前向仍需生成掩码并承担一定计算开销。

## 6. 实用指南

论文提供代码：`github.com/alexmanoo/ternary_adaptation`。复现时需：准备三值骨干；对注意力和 FFN 矩阵插入适配器；按层维度选择尽可能均衡的 \(p,q,r,s\)；使用 FP32 代理、STE 和 AdamW。论文设置学习率为 \(1.5\times10^{-3}\)（PTQ骨干）或 \(10^{-4}\)（预训练三值模型），训练一轮，batch size 16；推荐 Balanced 初始化。迁移到分类、生成或数学推理任务时，只需替换数据集和任务头，并冻结三值骨干，仅训练两个代理因子。

## 7. 总结

**核心思想：用三值乘法掩码适配三值权重。**

**速记版 Pipeline：**
1. 把模型权重压成 -1、0、1。  
2. 用两个小三值矩阵拼出完整修改图案。  
3. 用浮点副本训练，前向时强制变回三值。  
4. 训练后逐元素合并，直接得到新的三值模型。

**Key Findings:**

- We propose ternary multiplicative adaptation, which represents discrete updates of ternary weights such as sign flips or zeroing through a low-rank Kronecker factorization into two small ternary matrices applied element-wise to ternary weights.
- Experiments on six models across language and vision, including ternarized LLaMA-3 1B and 3B and a ternary ViT-B/16, demonstrate that our method recovers much of the performance lost to quantization and outperforms strong low-bit and ternary baselines.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24469v1)
- [arXiv](https://arxiv.org/abs/2608.24469v1)

---

<a id='2608.24365v1'></a>
## [MaST: Motion-aware Sparse Pipeline for Lightweight Object Tracking](https://arxiv.org/abs/2608.24365v1)

**Authors:** Qingmao Wei, Fagui Liu, Dengke Zhang, Qingze He, Quan Tang

**Published:** 2026-08-25

**Categories:** cs.CV

**Abstract:**

Transformer-based object trackers are renowned for their strong performance, yet dense token processing often leads to prohibitive computational cost, limiting real-time deployment on edge devices. While recent works explore token pruning to reduce computation, they often stop short of an end-to-end sparse pipeline, as early-layer token scores can be noisy without a motion prior, and many trackers ultimately fall back to dense reshaping to feed the dense prediction head that partially negates the savings.   We introduce Motion-aware Sparse Tracker (MaST), a sparse tracking framework that makes sparsity effective from tokens to boxes. First, MaST injects a lightweight motion prior to refine cross-attention-based importance scores, enabling earlier and more stable token reduction in the search region. Second, we introduce a natively sparse prediction head that operates directly on the retained unstructured tokens with a score-first, regress-once design, eliminating dense padding/reshaping and reducing redundant computation.   Extensive experiments on multiple benchmarks demonstrate that MaST establishes new state of the art among lightweight trackers, where MaST-tiny attains 63.8 AUC on LaSOT and 80.1 SUC on TrackingNet, surpassing the prior best AsymTrack-S by +1.0 AUC and +2.2 SUC   while running at 152 FPS on Jetson Nano, nearly twice as fast as AsymTrack-S at 88 FPS. Code is available at https://github.com/TsingWei/MaST.

**Analysis:**

## 1. 摘要翻译

基于Transformer的目标跟踪器性能强大，但密集Token处理带来高昂计算成本，限制了其在边缘设备上的实时部署。近期方法虽尝试剪枝Token，却通常未形成端到端稀疏流程：一方面，早期层的Token重要性分数受噪声影响较大，缺乏运动先验；另一方面，许多方法最终仍需将稀疏Token重新填充、变形为密集特征图，以输入密集预测头，部分抵消了加速收益。本文提出运动感知稀疏跟踪器MaST，使稀疏性从Token延伸至边界框预测。首先，利用轻量级运动先验修正基于交叉注意力的重要性分数，使搜索区域能够更早、更稳定地削减Token。其次，提出原生稀疏预测头，直接处理保留的非结构化Token，并采用“先评分、后回归一次”的策略，避免密集填充与重复计算。实验表明，MaST-tiny在LaSOT上达到63.8 AUC、TrackingNet上达到80.1 SUC，并在Jetson Nano上以152 FPS运行。

## 2. 方法动机分析

**驱动力**：Transformer跟踪器的主要瓶颈不是参数量，而是长Token序列上的注意力计算，尤其是搜索区域Token较多。作者希望同时解决“编码器算得多”和“预测头仍按密集方式算”两个问题。

**现有痛点**：  
1. 早期交叉注意力较弥散，直接据此剪枝会误删目标Token，因此OSTrack等方法往往推迟到中后层剪枝，前几层仍是全量计算。  
2. 注意力或辅助预测器主要反映当前帧外观，未利用目标跨帧运动连续性。  
3. 稀疏Token经过填充和二维重排后进入卷积头，空位置仍参与计算；即使使用MLP，也常对所有位置进行回归。

**核心假设**：早期剪枝本身并非必然损害精度，关键在于能否用上一帧位置提供可靠的空间筛选依据；若保留局部锚点式框表示，预测头也可以直接在非结构化Token上工作。

## 3. 方法设计详解

### 3.1 整体流程

输入模板图像 \(Z\) 和搜索区域 \(X\)，采用16×16图像块嵌入并拼接为Token序列。经过第一个Transformer块后，在搜索Token上执行一次稀疏化，随后所有Transformer层只处理保留Token，最后由稀疏预测头输出目标框。

### 3.2 运动感知Token筛选

首先计算搜索Token与模板中心Token的交叉注意力。设搜索查询为 \(Q_x\)，模板中心键为 \(k_c\)，第 \(i\) 个搜索Token的外观重要性为：

\[
s_i=\frac{\exp(q_i^\top k_c/\sqrt d)}
{\sum_k\exp(q_k^\top k_c/\sqrt d)}.
\]

该设计不聚合全部模板Token，而只使用模板中心Token，原因是模板通常以目标为中心，中心Token更可能提供纯净前景信息。

然后根据上一帧预测框 \(b_{t-1}=(x,y,w,h)\)构造二维高斯运动窗：

\[
G_t(u,v)=\exp[-(u-x)^2/(2\sigma_x^2)-(v-y)^2/(2\sigma_y^2)],
\]

其中 \(\sigma_x=\gamma w,\sigma_y=\gamma h\)，实验中 \(\gamma=0.5\)。最终分数为：

\[
w_i=G_t(u_i,v_i)s_i.
\]

作者不是硬性限制搜索区域，而是用高斯窗软重加权：靠近上一位置的Token优先级提高，但外观证据足够强的远处Token仍可能被保留。随后固定选取Top-K搜索Token；默认保留30%，并保存每个Token原始二维网格坐标。

### 3.3 原生稀疏预测头

保留Token直接输入两个轻量MLP分支：

- **评分分支**：为每个Token预测置信度 \(s_k\)，通过  
  \[
  k^*=\arg\max_k s_k
  \]
  选出最可能的目标Token。
- **回归分支**：只对选中的Token执行一次框回归，预测相对其网格坐标的偏移和宽高：
  \[
  \hat b=(u_{k^*}+\delta_x,v_{k^*}+\delta_y,w_{k^*},h_{k^*}).
  \]

因此，评分复杂度随保留Token数增长，而回归只进行一次。它保留了传统局部锚点框定义，却删除了密集重排、空位置卷积以及所有位置重复回归。训练时，分类损失作用于稀疏Token；回归监督选择距离真实中心最近的保留Token，并使用L1与GIoU损失。

## 4. 方法对比与创新

MaST与OSTrack的根本区别在于：OSTrack主要依赖交叉注意力并较晚剪枝，MaST将“上一帧位置”引入早期筛选；与动态退出方法不同，MaST采用固定Top-K预算，延迟更稳定；与传统稀疏主干不同，它同时改造预测头，形成端到端稀疏链路。

主要创新包括：  
1. **运动先验与外观注意力乘法融合**，使第一层即可可靠剪枝；  
2. **稀疏Token原生预测头**，避免恢复密集网格；  
3. **先评分、后单次回归**，进一步去除冗余预测。

最适用于目标运动连续、边缘算力受限、需要稳定帧率的场景，如无人机、机器人和移动端跟踪。

## 5. 实验分析

作者在LaSOT、TrackingNet、GOT-10k、VastTrack及多个UAV数据集上比较，并进行筛选策略、预测头、剪枝层和保留率消融。代表性结论是：MaST-tiny达到LaSOT 63.8 AUC、TrackingNet 80.1 SUC，并在Jetson Nano上达到152 FPS；仅注意力剪枝精度明显下降，而“注意力+运动窗”几乎恢复密集模型精度。其不足是局部搜索范式仍限制大幅运动和长时间遮挡，且高分辨率输入在剪枝前仍需承担较多计算。

## 6. 实用指南

论文提供代码仓库：`github.com/TsingWei/MaST`。复现时需使用ViT-Tiny与MAE-lite初始化，模板/搜索尺寸默认为128/256，Patch为16，保留率30%，第一层剪枝；先训练300轮密集模型，再启用稀疏化微调50轮，前10轮将保留率从100%线性降至30%。训练中运动窗中心可直接设为搜索图中心，测试时改为上一帧预测框；采用AdamW、权重衰减 \(10^{-4}\)，骨干和新增模块学习率分别为 \(4\times10^{-5}\) 与 \(4\times10^{-4}\)。该思想可迁移到检测、分割或视频理解：只要任务具有稳定的历史位置/轨迹先验，并将密集输出头改写为“稀疏候选评分+少量候选精确预测”即可。

## 7. 总结

**核心思想：用运动先验引导端到端稀疏跟踪。**

**速记版Pipeline：**  
1. 模板和搜索图切成Token并进入Transformer。  
2. 用模板外观分数结合上一帧位置，优先保留可能属于目标的Token。  
3. 后续编码层只处理这些Token。  
4. 对保留Token逐一打分，只对最高分位置回归一次边界框。

**Key Findings:**

- We introduce Motion-aware Sparse Tracker (MaST), a sparse tracking framework that makes sparsity effective from tokens to boxes.
- Second, we introduce a natively sparse prediction head that operates directly on the retained unstructured tokens with a score-first, regress-once design, eliminating dense padding/reshaping and reducing redundant computation.
- Extensive experiments on multiple benchmarks demonstrate that MaST establishes new state of the art among lightweight trackers, where MaST-tiny attains 63.8 AUC on LaSOT and 80.1 SUC on TrackingNet, surpassing the prior best AsymTrack-S by +1.0 AUC and +2.2 SUC   while running at 152 FPS on Jetson Nano, nearly twice as fast as AsymTrack-S at 88 FPS.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.24365v1)
- [arXiv](https://arxiv.org/abs/2608.24365v1)

---

