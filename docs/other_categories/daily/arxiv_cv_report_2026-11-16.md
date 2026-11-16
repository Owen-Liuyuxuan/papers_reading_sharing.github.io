time: 20261116

# Arxiv Computer Vision Papers - 2026-11-16

## Executive Summary

## 今日研究速览

本期十篇论文呈现出三条相互关联的路线：一是让表征和训练目标更贴近任务结构，包括AnisoFlow按数据方向学习流模型预测目标、V-JEPA-LM用成对反事实监督读取视频运动，以及CLIP层级角分离研究解释图文嵌入如何传递概念树结构；二是为视觉系统建立更可检验的机制与评估，例如LASR区分世界模型潜变量是否真正支持动作，RAWATTA按多视图注意力层的几何职责选择近似方式；三是把感知不确定性接到操作或场景后果上，涵盖触觉真实—仿真迁移、本体感觉锚定与闭环偏置、数字变焦造成的单目深度尺度错误，以及自动驾驶对地图交通流的系统影响。

值得优先精读的论文取决于研究目标。做生成模型或视频表示的读者可从AnisoFlow与V-JEPA-LM入手：前者在匹配JiT主干上报告ImageNet FID从1.90降到1.55，后者通过匹配监督对照显示反事实样本对运动问答的增益；两者也都给出清晰的适用边界。关注机器人学习的读者可读TouchTwin与本体感觉锚定研究，分别检查跨仿真触觉预测和训练噪声折扣能否外推到闭环。做三维视觉或多模态表征的读者可关注RAWATTA和CLIP层级传递：前者的加速同时依赖CUDA执行器且需要目标域校准，后者的理论保证依赖固定图像特征、均衡标签和足够大的扰动间隔，真实CLIP原型尚未满足其充分条件。

整体来看，本期趋势是从“模型是否预测得准”转向“内部表示是否可读、行动时是否可靠、评价是否揭示真实因果约束”。新方法常能通过结构化监督、可控变换或干预获得明确收益，但评估也更强调匹配对照、失败条件和外推范围。阅读时宜优先检查训练数据与测试构造是否独立、提升是否来自方法本身，以及仿真或简化理论与部署场景之间还缺哪些验证。

---

## Table of Contents

1. [Learning Anisotropic Prediction Targets for Flow Models](#2611.17767v1)
2. [V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations](#2611.17831v1)
3. [TouchTwin: Scaling Human Touch Data via Real-to-Sim-to-Real](#2611.17619v1)
4. [Ego2Flow: Benchmarking the Traffic-System Impact of Autonomous Driving on Maps Distilled from Real-World Roads](#2611.17720v1)
5. [VACO: Vertex-Aligned Decoding for Dense Hand Contact Estimation from a Single Image](#2611.17755v1)
6. [Beyond Prediction Gain: LASR as a Metric for Action-Grounded Latents in World Models](#2611.17775v1)
7. [Provable Cross-Modal Transfer of Hierarchical Concept Structure in CLIP Embeddings](#2611.17671v1)
8. [Training-Free Acceleration of Multi-View 3D Transformers via Regime-Aware Attention Approximation](#2611.17629v1)
9. [Zoom In, Scale Out: Metric Depth Models Mistake Digital Zoom for Moving Closer](#2611.17903v1)
10. [Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost](#2611.17658v1)

---

## Papers

<a id='2611.17767v1'></a>
## [Learning Anisotropic Prediction Targets for Flow Models](https://arxiv.org/abs/2611.17767v1)

**Authors:** Rayhan Zirvi, Hansheng Chen, Gordon Wetzstein, Leonidas Guibas

**Published:** 2026-11-13

**Categories:** cs.CV, cs.LG

**Abstract:**

Flow-based generative models are trained on a hand-chosen prediction target, such as the velocity, the noise, or the clean data. This choice strongly affects training and sample quality. We introduce Anisotropic Flow Models (AnisoFlow), which replaces a fixed prediction target with a fully learnable one. AnisoFlow learns to scale the noise component of the target anisotropically, assigning a separate nonnegative weight to each direction in a jointly learned basis. A closed-form map recovers the velocity from the network's prediction, leaving the architecture, flow-matching loss, and sampler unchanged. Since recovery is exact, training optimizes the target purely to make it easier for the network to predict. We theoretically analyze the resulting prediction error and predict that the preferred noise weight in each direction scales near-linearly with data variance, in agreement with the weights learned empirically in pixel space. AnisoFlow improves FID from 1.90 to 1.55 over the JiT baseline on ImageNet 256$\times$256, by learning only the prediction target, without representation alignment or architectural changes. In pixel-space text-to-image generation, AnisoFlow improves image quality and, through better-conditioned velocity recovery, largely removes patch-seam artifacts under reduced-precision inference.

### 论文解读

#### 摘要翻译
扩散或流匹配模型需先选预测目标，例如噪声、速度或干净图像。AnisoFlow不固定这个选择，而是在学习的正交方向基中，为各方向分别学习噪声权重，再用闭式公式还原速度。模型架构、流匹配目标和采样器都不变。在ImageNet像素生成上，JiT-H/16的FID由1.90降至1.55；像素文本生图在多个指标改善，低精度推理也更稳定。

#### 方法动机分析
这些参数化理想情况下携带相同信息，却对应不同回归难度，这是固定参数化的不足。像素patch包含许多方差不同的坐标：统一预测速度会令某些方向的噪声难以估计；干净图像预测则可能在低噪声时让速度恢复数值不稳定。作者希望根据数据和网络自身误差，为每个方向找到折中权重。理论并非完整训练动力学，而是一个局部误差模型，预测权重与数据方差近似成比例。

#### 方法设计详解
对数据x和噪声ε，模型目标写为Wε−x，W是正半定矩阵。将W分解为正交基与非负特征值后，每个方向都能单独调节噪声权重。主干网络读取加噪样本和时间步，输出新目标；闭式恢复模块把结果转换为标准速度，再计算原来的速度误差并交给原采样器。权重通过指数参数化和Adam学习；正交基默认通过预测误差的滑动矩阵做特征分解更新。初始化可以从单位阵开始，也可使用图像patch的PCA。像素patch含768维，基与权重在空间位置间共享；推理时冻结学得的基，不再求特征分解。

#### 方法对比分析
相比固定噪声、速度或图像目标，AnisoFlow允许每个方向不同权重；相比AsymFlow二值掩码，它允许连续调节；相比改变主干或加入表示对齐，它只改目标参数化。匹配JiT-H/16实验中，连续权重配PCA基已优于二值AsymFlow，最终学习基与权重的谱更新得到更低FID。这个结果支持目标本身能带来收益，但更复杂的优化增加训练状态，且基更新依赖预测误差矩结构假设。

#### 实验分析（精简版）
ImageNet 256×256上，600轮JiT-H/16 FID为1.90，AnisoFlow为1.55；网络参数约953M不变，推理从0.179增至0.180秒/步。文本生图匹配训练数据和设置后，HPSv3从7.71升至8.03，COCO-10k FID从22.0降到20.1，patch FID从24.4降到22.7；GenEval则由0.820轻微降至0.814。潜空间SiT/RAEv2表现与默认目标接近。作者还显示低精度量化时基线产生接缝或噪声，AnisoFlow较稳定；这不等于完整的低精度定量评估。

#### 实用指南
适合像素空间生成、预测目标研究或关心量化稳定性的研究者。复现时固定骨干、数据、采样器和评测协议，只替换目标；报告各项生成指标，避免只挑改善项。训练需维护D×D正交基和矩阵统计，谱分解每步执行；推理虽不分解，仍有基变换成本。若移到潜空间，应先确认收益是否超过基线，因为论文中的改进较小。

#### 总结
AnisoFlow让模型学习每个数据方向的预测目标，使高方差与低方差坐标获得不同噪声权重。它在像素生成中取得明确提升，但理论有条件，文本指标有例外，潜空间收益有限。

1. 沿正交数据方向设置独立噪声权重。
2. 联合优化目标权重、方向基与生成网络。
3. 恢复速度并使用标准流匹配损失与采样器。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17767v1)
- [arXiv](https://arxiv.org/abs/2611.17767v1)

---

<a id='2611.17831v1'></a>
## [V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations](https://arxiv.org/abs/2611.17831v1)

**Authors:** Keuntae Kim, Dongjun Kim, Yong Suk Choi

**Published:** 2026-11-13

**Categories:** cs.CV, cs.AI

**Abstract:**

World-model video encoders such as V-JEPA learn how scenes change (direction, speed, order, state) from video alone. To answer questions, they are joined to a language model trained on captions and question-answer pairs. But each caption or question names only a few attributes of a clip, so the model learns to retrieve those attributes rather than to read the clip, and the dynamics the encoder carries go unused: on the same items, a linear probe reads speed from frozen V-JEPA features at 87% while the VLM answers at chance. We show where dynamics lives in a frozen encoder: in a subspace shared across videos for predictive encoders such as V-JEPA, but not as a shared component of frame-wise, text-aligned encoders such as SigLIP. We then build counterfactual supervision that makes the language model read it: pixel transformations turn public videos into answer-balanced variants whose answers are set by the transformation and differ only in the change. Trained in a single stage on 4.53M public samples with 32 visual tokens per frame, where open video LLMs use several stages, 9M to 172M samples and 64 to 256 tokens per frame on average, the resulting V-JEPA-LM models (1B to 8B, Qwen3 and Llama) lead every size class in average score across eleven video benchmarks; our 4B model scores 72.1, against 61.1 for the best open 4B and 64.8 for the best open 8B model. A controlled encoder x supervision study shows that the encoder decides what can be read and the supervision what is read: at a matched budget, counterfactual supervision adds 4.5 points on average and 38 on the speed questions of natural benchmarks. Code and models: https://github.com/kkt94/Llama-V-JEPA-LM.

### 论文解读

#### 摘要翻译
视频世界模型编码器如V-JEPA能从画面变化中学到方向、速度与事件顺序，但后接语言模型通常只用字幕和问答训练。论文观察到，冻结V-JEPA特征中的速度可被线性探针以约87%准确率读出，而常规视频语言模型在相同问题上接近机会水平。作者提出让模型利用已有动态信息的反事实监督，并在11个视频基准上报告不同模型规模的结果。

#### 方法动机分析
普通字幕通常只说场景中少数属性，例如“桌上有球”，而不描述球如何移动。若训练问题与答案没有明确要求关注时间变化，语言模型可能依赖静态内容，浪费预测式视频编码器保留的动态信息，是常规监督的瓶颈。论文的关键设计是让两条训练样本共享同一片段语义和同一问题，只通过可控的像素变换改变运动，并按规则翻转答案。这样可削弱共有内容捷径，使监督集中在变换相关的运动差异上。

#### 方法设计详解
作者从公开视频生成14类反事实构造，覆盖速度、方向、顺序和状态等变化；共用33个来源，形成509,193行，占训练抽样的6.11%。变换按像素和规则完成，不依赖人工为每个样本编写标签。视频取32帧，经冻结的V-JEPA 2.1编码器提取时空特征，再由bridge压缩为1,024个token，交给自回归语言模型回答。整个流程的输入是32帧视频及问题，输出是语言答案；视觉encoder冻结，bridge与语言模型在单阶段训练。字幕与常规问答仍是主体监督，反事实行以平衡的成对形式加入训练。推理时输入同样的32帧视频和文本提示，模型自回归生成答案。作者测试1B至8B模型，并设置监督消融、时间控制、保留变换测试和子空间干预。

#### 方法对比分析
相比再收集大量自然问答，该方法从已有公开视频程序化生成变化标签；相比只加普通时间问答，它明确构造同内容、异答案配对；相比逐帧文本对齐编码器，V-JEPA的时空token具有跨视频共享的变化子空间，提供更适合读取的信号。论文的匹配实验支持编码器与监督各自起作用：只有监督改变而其余训练计划固定时，反事实成对数据带来收益。生成式标注降低人工成本，但有效性依赖变换与答案规则正确，且构造分布不能覆盖所有真实运动场景。

#### 实验分析（精简版）
最直接的对照是2B模型、匹配编码器和训练计划。ViT-L在8k步时，十二项平均从仅字幕问答的56.65提升至反事实监督的61.18；速度项从32.4提升至65.3，六个保留构造集合的平均为91.3。论文报告箭头时间替代信号未达到同等收益。最终4B模型在11个基准十二项平均为72.14；统一评测下最佳开放4B与8B对照分别为61.08和64.79。增益主要来自运动类能力，知识类项目大致持平或略有下降。每个公开模型只训练一次，跨论文比较受实现差异影响。

#### 实用指南
复现时需确保变换、问题和答案规则成对一致，并将变换家族或来源留出，防止只测熟悉模板。论文模型使用32帧视频，4B模型训练报告499 H100小时；评测覆盖也影响结论。将方法用于机器人视频时，可尝试时间缩放或方向相关问答，但真实接触和动力学可能不满足简单像素变换规则，需先人工审计，再做闭环验证。该迁移建议是基于方法结构的推断。

#### 总结
论文表明，视频模型不只是需要更大数据量，也需要能指向目标表征的监督。共享场景、改变运动答案的成对训练样本，能让语言模型更好读取冻结V-JEPA中的动态信息；优势集中在运动理解。32帧范围、单次训练、显著算力成本及构造类型，仍限制其外推范围。

1. 冻结视频编码器，从32帧片段提取时空表示。
2. 对视频做可控变换，保持问题并按规则改变答案。
3. 混合成对反事实数据与常规监督，训练语言模型读取动态线索。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17831v1)
- [arXiv](https://arxiv.org/abs/2611.17831v1)

---

<a id='2611.17619v1'></a>
## [TouchTwin: Scaling Human Touch Data via Real-to-Sim-to-Real](https://arxiv.org/abs/2611.17619v1)

**Authors:** Huiwen Han, Nuo Chen, Ziyao Zeng, Dayou Li, Hao Wang, Amir Hossein Rahmati, Xiaoning Qian, Bangya Liu, Zhiwen Fan

**Published:** 2026-11-13

**Categories:** cs.RO

**Abstract:**

Human demonstrations are becoming the main source of data for dexterous manipulation, and real-to-sim-to-real pipelines multiply each demonstration in simulation. These pipelines rebuild the interaction from vision, which recovers how the hand moves. Which fingers touch, when, and how hard remain unobserved, because a hand pose shows where each finger stops on the surface rather than how far it is driven to close. This gap is a force gap: on held-out recordings, a hand policy that imitates the observed pose grips with 0.31x the recorded force, and a policy trained on physics replays of vision-tracked poses grips with 1.55x. TouchTwin closes the gap with measured human touch. Each selected TouchScale recording, captured with head and wrist RGB-D cameras and an 880-taxel glove, is rebuilt in MuJoCo. Coding agents write the objects as physics-tested programs, and a per-subject tactile hand replays the interaction with the recorded touch setting the contact and the closing force of each finger. In simulation, a policy trained on touch-grounded replays grips at 0.99x the recorded force and succeeds in 20/20 trials on held-out recordings and 35/36 unseen perturbations; a contact predictor trained on them misses 13% of real contact frames, against 32% for vision-based replays. Each replay also seeds perturbed demonstrations with simulated contact and force labels, rendered as realistic video in new views and scenes.

### 论文解读
#### 摘要翻译
TouchTwin 研究如何把人类灵巧操作演示转换成可扩增的物理仿真数据。传统视觉重建能够估计手的运动，却难以确定被遮挡的接触位置、接触时刻和抓握力。作者把头戴及腕部 RGB-D 影像与 880 个触觉单元的手套记录结合，在 MuJoCo 中重建手物交互，并利用真实触觉约束模拟接触。实验显示，触觉校准的重放更接近真实抓力，也改善了接触预测和模拟策略表现；每段重放还可用于扰动条件并生成新视角视频。

#### 方法动机分析
手指停在物体表面的可见位置，并不能说明手指施加了多大压力。若只拟合可见姿态，策略可能抓得过轻；若由视觉轨迹建立物理仿真，接触求解又可能产生过大的抓力。TouchTwin 的关键假设是：真实触觉提供视觉中不可辨识的接触与力信息，因此可作为仿真校准信号，而非仅作为模型需要从图像预测的标签。

#### 方法设计详解
这个 input-to-output pipeline 的输入是头戴与腕部 RGB-D、手部姿态以及 880 个 taxel 的触觉序列。第一步为每位参与者拟合带关节链和传感器布局的个体手模型，并把物体写成可执行的 MuJoCo 程序；对象的活动部件也要能被接触正确驱动。第二步将视觉追踪的手部运动放入场景，但不直接把表面姿态当成完整抓握：真实 glove 信号被映射到模拟手的 taxel，约束每根手指何时接触、接触在哪里以及闭合到多大力。第三步运行物理重放，由接触而非脚本推动物体，再通过检查代理对照模拟渲染与原始视频、以及手套力曲线；不一致时返回修复对象程序或交互设置。第四步固定其余条件，逐项扰动抓姿、搬运动作、手物摩擦、物体质量或摇晃，记录滑移、掉落、接触和力。第五步把重放状态交给 Wan2.2 视频解码器，生成新视角或场景，同时继承模拟的触觉与力标签。

#### 方法对比分析
视觉重放只复现观察到的运动，不能决定隐藏的压力；TouchTwin 将触觉直接放进接触动力学，因此输出可用于物理扰动的交互状态。与从视频预测触觉的路线相比，这里需要真实触觉记录，但可在重放时改变条件并得到新的力标签。它更适合具备触觉示范、且需要分析抓握余量的操作数据扩增。

#### 实验分析（精简版）
在留出的真实第一视角记录上，触觉重放训练的接触预测器 F1 为 0.890，视觉重放为 0.778；漏检率分别为 13.1% 和 32.2%。手策略的抓力达到记录值的 0.99 倍，留出记录成功率为 20/20；姿态模仿基线为 0.31 倍和 17/20。结果主要来自模拟策略与少量操作类别，不能据此推断真实机器人部署效果。

#### 实用指南
复现需要成对的 RGB-D 与触觉手套数据、逐对象物理资产、参与者手模型和 MuJoCo 重放管线。论文的策略训练采用 N0-VTLA，视频推理使用 Wan2.2 将仿真渲染为新视角；具体复现应先核对触觉到模拟 taxel 的标定、对象摩擦和质量等参数。将该思路迁移到机器人数据时，需重新验证触觉传感器与机器手的接触映射及真实闭环策略。

#### 总结
核心思想：实测触觉校准仿真接触
1. 采集手部影像、运动和触觉。
2. 重建可交互的手与物体模型。
3. 用真实触觉匹配模拟接触和抓力。
4. 扰动物理条件并生成带标签数据。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17619v1)
- [arXiv](https://arxiv.org/abs/2611.17619v1)

---

<a id='2611.17720v1'></a>
## [Ego2Flow: Benchmarking the Traffic-System Impact of Autonomous Driving on Maps Distilled from Real-World Roads](https://arxiv.org/abs/2611.17720v1)

**Authors:** Yu Dong, Weiying Shen, Hangyu Li, Shu Han, Hong Li, Xiangbo Gao, Keshu Wu, Zhengzhong Tu, Soyoung Ahn, Zhiyuan Liu, Xiaopeng Li

**Published:** 2026-11-13

**Categories:** cs.RO, cs.CV

**Abstract:**

End-to-end autonomous driving is predominantly evaluated through ego-vehicle safety and task completion. Its deployment in urban traffic, however, creates sustained interactions among vehicles whose collective effects remain insufficiently evaluated. We introduce Ego2Flow, a closed-loop benchmark for assessing the traffic-system impact of autonomous driving on maps distilled from real-world cities. To construct realistic 3D urban environments for sustained closed-loop traffic simulation, we develop an agent-guided map distillation framework that integrates real-world maps and observations with complementary geographic information. Through an iterative feedback mechanism and a traffic information graph that jointly guide scene generation, the framework preserves lane topology, critical road geometry and decision-relevant visual conditions while simplifying nonessential details and completing unobserved regions. Using our proposed framework, we generate and publicly release six CARLA maps from the Argoverse 2 dataset, covering the corresponding real-world regions and enabling paired comparisons of driving behavior under real and simulated observations. On these maps, Ego2Flow evaluates how autonomous driving policies affect traffic efficiency, safety, stability and energy consumption across diverse road structures. Compared with existing construction methods, our framework achieves the lowest average displacement error between planning outputs from paired real and simulated observations. We further benchmark representative driving policies to characterize their system-level impact on urban traffic. Project Page: https://github.com/Fisher-007/Ego2Flow.

### 论文解读
#### 摘要翻译
Ego2Flow提出一种闭环基准，用于衡量自动驾驶策略对周围交通系统的影响。作者从 Argoverse 2 真实道路记录构建六张 CARLA 城市地图，并设计智能体引导的地图蒸馏方法，保留道路拓扑、关键几何和决策相关视觉条件。基准覆盖交通效率、安全、稳定性和估算能耗，比较不同驾驶策略在持续交通、瓶颈和扰动传播中的整体表现。

#### 方法动机分析
单车安全到达不等于交通系统表现良好。制动可能沿车队传播，造成延迟或速度波动；而城市部署会让策略与许多车辆持续交互。作者的核心假设是，只要仿真地图保留影响驾驶决策的道路和视觉因素，就可以在可控闭环环境中观察策略造成的系统级影响。

#### 方法设计详解
输入包括 Argoverse 2 的车道图、相机、LiDAR、位姿和目标标注，以及 OpenStreetMap 路网和地形。地图构建智能体以交通信息图组织道路元素、证据、任务和仿真实现之间的关系，并据此选择保留、简化、增强或补证。对道路缺口，方法只对可编辑区域的潜变量加噪去噪，每一步用掩码恢复保留车道；组装时继续约束端点、切线、宽度和高程，检查自交及与建筑的冲突。外观候选依据源图生成，再装配到 CARLA。静态审查比较来源与渲染的道路、标志和可见性；动态审查运行路线、驾驶策略和对真实轨迹的回放。问题定位后反馈给智能体，触发局部修订。基准在六城运行持续交通、瓶颈、扰动传播三类任务，分别设置全自动驾驶和约三成自动驾驶的混合交通，并计算效率、安全、稳定和模型估算能耗。

#### 方法对比分析
常见驾驶评测聚焦自车成功率，交通控制研究则常以显式交通状态为输入。Ego2Flow将传感器驱动策略置于真实道路蒸馏出的三维场景，评价它对周围交通的影响；地图方法优先保持与规划有关的结构，而非重建所有外观细节。总分便于概览，但规则控制器与端到端策略输入不同，分项指标更适合判断优劣来源。

#### 实验分析（精简版）
在六城、5 秒规划预测对比中，Ego2Flow 的平均 ADE 为 0.7966 米，比 Cam2Sim 的 0.8784 米低 9.3%。完整地图框架 ADE/FDE 为 0.3465/0.9285 米，去掉静态反馈后为 0.3706/0.9918 米。交通策略综合分中 IDM 为 71.804，LEAD 为 54.215。地图误差使用冻结模型对真实和渲染观测的规划差作代理，能耗来自物理模型估算，不能直接视作真实道路部署结果。

#### 实用指南
作者公开六张 CARLA 城市地图。复现时需使用 CARLA 0.9.16，闭环推理采用同步 0.05 秒步长，并保持相同车辆物理参数、请求路线和预热过程。任务区域与流量配置会影响系统指标，建议同时报告八个分项和失败比例。迁移到新的城市需重新对齐地图、相机与轨迹，再做静态和动态审查。

#### 总结
核心思想：从单车评测走向交通系统
1. 融合道路观测和地理数据。
2. 蒸馏并审查可闭环运行的城市地图。
3. 让多车策略交互并测量系统影响。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17720v1)
- [arXiv](https://arxiv.org/abs/2611.17720v1)

---

<a id='2611.17755v1'></a>
## [VACO: Vertex-Aligned Decoding for Dense Hand Contact Estimation from a Single Image](https://arxiv.org/abs/2611.17755v1)

**Authors:** Binhong Ye, Jin-Chuan Shi, Tao Liu, Xin Wang, Hao Chen, Chunhua Shen

**Published:** 2026-11-13

**Categories:** cs.CV

**Abstract:**

Dense hand contact estimation from a single RGB image is a foundational primitive for dexterous manipulation and 3D interaction analysis. Direct predictors typically decode the entire per-vertex contact map from one shared query, leaving individual vertices ungrounded in the image and smearing contact across adjacent, non-touching surfaces. We present VACO, a vertex-aligned decoder in which each MANO vertex samples local image features at its projected location. A hand-reconstruction backbone, jointly trained for reconstruction and contact, predicts these locations at inference without an auxiliary reconstructor. Training uses noise-perturbed ground-truth projections and end-to-end feature adaptation to improve robustness and contact-sensitive features. A single VACO checkpoint sets a new state of the art on five benchmarks spanning hand-object, hand-human, hand-scene, and bimanual interactions. It improves F1 by +20.7 pp over the strongest prior direct predictor on Hi4D and generalizes to the held-out HOT3D and HOCap, with gains of +5.5 and +2.7 pp. The resulting contact predictions also improve downstream HOI reconstruction.

### 论文解读
#### 摘要翻译
VACO从单张RGB图像估计手部网格每个顶点是否接触。它让每个顶点依据自身投影位置读取局部图像特征，再结合手部整体结构和场景上下文判断接触。一个模型在五类手部交互基准上取得领先，并改善后续手物重建。
#### 方法动机分析
重建后按几何距离判接触会把手、物体重建误差层层传递；直接回归虽快，却常用一个共享查询生成整张接触图，模型不知道哪个顶点对应图像的哪个位置，易把接触扩散到邻近手指。VACO的假设是：给每个网格顶点对齐局部视觉证据，能更准确地区分真正接触的表面，同时保留全局交互信息。
#### 方法设计详解
输入图像先经WiLoR骨干得到ViT特征和MANO手网格。模型将网格顶点投影到图像平面，从三种分辨率的特征图双线性采样；每份局部特征形成一个带顶点身份编码的token。四层Transformer先在顶点间做自注意力，表达握拳或捏取等结构关系，再对原图特征做交叉注意力，补入物体与遮挡上下文，最后逐顶点输出接触概率。训练时使用标注投影并加入位置噪声，推理时改用联合训练的手部重建结果；多尺度网格与关节读出、类别平衡损失及重建目标共同稳定学习。训练使用约62.6万帧、11个数据集，10轮、有效批量64、8张RTX 4090，约需768 GPU小时。
#### 方法对比分析
相较重建后查询，VACO不依赖完整物体网格；相较共享查询式直接预测，它用固定网格拓扑建立逐顶点图像锚点。匹配骨干实验显示，顶点对齐解码仍能提升五基准平均F1。其创新贡献是把几何对应从重建后的距离计算改成直接视觉接地；与共享查询的区别是每个顶点有独立采样位置。匹配HaMeR骨干、训练数据和损失的对照中，VACO平均F1为0.677，HACO为0.622，支持提升来自解码方式而非仅换用更强特征骨干。它适合从单图获得细粒度接触先验，不能直接提供力、深度或时间连续性。
#### 实验分析（精简版）
Hi4D上VACO的F1为0.860，HACO为0.653；未见的HOT3D和HOCap分别提升5.5与2.7个百分点。五基准平均F1为0.699，去掉编码器微调后降至0.486，说明接地特征学习贡献显著。下游HO3D-v3的右手Chamfer距离由23.11±22.77毫米降至12.46±8.65毫米，但该项仅测四段序列。指标为单次实验，且标签多数依赖距离阈值。
#### 实用指南
方法基于PyTorch、WiLoR和MANO；推理约100毫秒/图，训练成本较高。复现需统一顶点接触标注、投影相机和左手镜像索引，并谨慎处理遮挡。机器人迁移可把顶点局部采样思想用于触觉单元，但仍需实体触觉和闭环实验验证。
#### 总结
核心思想：每个手部顶点看自己的图像证据
1. 图像预测手部网格与特征。
2. 顶点投影后采样多尺度视觉信息。
3. Transformer融合局部接触线索、手部结构和场景上下文。
4. 输出逐顶点接触，并作为重建先验。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17755v1)
- [arXiv](https://arxiv.org/abs/2611.17755v1)

---

<a id='2611.17775v1'></a>
## [Beyond Prediction Gain: LASR as a Metric for Action-Grounded Latents in World Models](https://arxiv.org/abs/2611.17775v1)

**Authors:** Yu Zhou, Changsong Lei, Yixiang Wei, Kairui Wen, Huankun Sheng, Ming Li, Yeying Fan, Yu-Hui Wen, Yong-Jin Liu

**Published:** 2026-11-13

**Categories:** cs.RO, cs.LG

**Abstract:**

World models that support planning, control, and interpretable reasoning require latent representations that genuinely encode action. Yet the large-scale video data from which such models are learned overwhelmingly lack action annotations. Latent-action models infer action-like variables from observation sequences, enabling action-aware learning from unlabeled videos or trajectories. A fundamental concern, however, remains: a latent that improves future prediction may simply encode the current state, episode progress, task identity, or other predictive shortcuts. Existing evaluation practices---reconstruction loss, rollout fidelity, action recovery, or downstream performance---cannot isolate this distinction. We introduce the Latent Action Support Ratio (LASR), a dimensionless metric that measures, under offline evaluation on held-out data, whether a learned latent provides predictive support for action-related targets beyond the current state and non-action alternatives. LASR provides two complementary summaries: GeoLASR captures typical support, while WorstLASR identifies the weakest case. We instantiate the LASR benchmark with recorded actions and corresponding physical state changes as targets, episode-held-out splits, one predictor type for all comparisons, and positive and negative controls. Experiments on recorded robot manipulation data and a matched multi-task panel demonstrate that LASR detects known action information, rejects representative non-action shortcuts, and enables graded comparisons among imperfect latent representations. We further use LASR to train and audit a lightweight residual adapter, obtaining descriptively better point estimates whose uncertainty is assessed with paired comparisons. LASR thus provides a principled offline protocol for measuring action-related predictive support in world-model latents.

### 论文解读
#### 摘要翻译
世界模型需要知道动作如何改变未来，但大量视频没有动作标注。潜变量即使让预测更准确，也可能只是记住当前状态、任务进度或场景线索。作者提出潜在动作支持比（LASR），在离线留出数据上比较潜变量、当前状态和非动作捷径对动作相关目标的预测能力，并用典型值GeoLASR与最差值WorstLASR汇总。
#### 方法动机分析
单看重建质量、rollout或下游成功无法区分“动作信息”与“轨迹本身容易预测”，也可能受预测器容量影响。LASR的核心假设是：若潜变量确实提供额外动作信息，在同一预测器和测试样本上，它应同时胜过状态基线与一个经验证集挑选的非动作控制。
#### 方法设计详解
对每个目标和数据折分别训练三条预测路径：只看当前状态S、看状态与潜变量(S,Z)、看状态与控制变量(S,b)。控制变量从进度、episode身份等候选项中按验证误差选出。三路都预测同一目标、使用相同模型族和损失，并在相同留出集估计风险。LASR=E_z/min(E_state,E_control)，所以低于1表示潜变量路同时胜过两个参照。跨目标与折的几何平均构成GeoLASR，最大单格构成WorstLASR；两者越低越好，Worst低于1才表示每格均通过。作者还给出条件Bayes风险分解，并证明仅由潜变量生成的确定性适配器不能凭空创造目标信息，只能让既有信息更容易被受限预测器读出。
#### 方法对比分析
相较单纯预测增益，LASR额外设置非动作负控，拒绝“状态虽解释不了、但进度可以解释”的假阳性；WorstLASR也能暴露平均值掩盖的失败折。它的创新在于把潜变量评估写成可校准、可比较的相对风险协议。指标依赖所声明的目标、控制集合和预测器，结果不能脱离协议解释。
#### 实验分析（精简版）
ALOHA静态电池数据取自49段演示，构成2048条对齐转移，按episode做五折留出，目标为记录动作与一步状态变化。DiLA的Geo/Worst为0.750/0.796，优于LaWAM的0.891/0.947；UniVLA的Worst为1.077，未通过全格判据。五任务匹配仿真面板上DiLA为0.688/0.719，但任务分层显示PegInsertionSide是明显失效例。结果是离线指标，不是策略成功保证。
#### 实用指南
复现需要同步状态、动作与后继观测，按episode隔离训练/验证/测试，并只在验证集选择捷径控制。报告每类目标及最差折；再用动作干预和实体或仿真闭环评测验证可执行性。代码工件包含评测和数据构造工具，但不含数据、权重、缓存或适配器训练循环。
#### 总结
核心思想：检验潜变量是否胜过动作捷径
1. 对齐状态、动作与未来变化目标。
2. 训练状态、潜变量、非动作控制三条匹配预测路径。
3. 用留出风险比及典型/最差摘要审计表征。
4. 通过干预和闭环执行继续验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17775v1)
- [arXiv](https://arxiv.org/abs/2611.17775v1)

---

<a id='2611.17671v1'></a>
## [Provable Cross-Modal Transfer of Hierarchical Concept Structure in CLIP Embeddings](https://arxiv.org/abs/2611.17671v1)

**Authors:** Nghia Nguyen, Tianjiao Ding, Lachlan Ewen MacDonald, René Vidal

**Published:** 2026-11-13

**Categories:** cs.CV, cs.LG

**Abstract:**

Contrastive vision-language models such as CLIP are trained to align images and text, not to represent taxonomies. Yet empirical studies show that CLIP embeddings exhibit hierarchical structure aligned with the hierarchy of semantic concepts. A theoretical characterization of why such hierarchy emerges from a non-hierarchical objective remains open. In this paper, we formalize this phenomenon as Hierarchical Angular Separation (HAS), a geometric property whereby each non-leaf concept embedding is more similar to its descendants than to off-branch concepts. Empirically, we show that HAS holds approximately in raw images and randomly initialized CLIP image embeddings, while in text embeddings it holds approximately only after contrastive learning. To theoretically explain this asymmetry, we analyze an unconstrained features model in which the image embeddings are fixed and only the text embeddings are trained. At the global minimizer, the text-side HAS violation percentage at each node is at most the image-side HAS violation and near-violation percentage. The key mechanism is the hierarchical co-occurrence structure of image-text pairs, which biases text embeddings toward root-centered image prototypes. For a linear contrastive loss, we derive a closed-form solution that makes this mechanism explicit. We extend the analysis to the InfoNCE loss and establish analogous guarantees under sufficient regularization. Synthetic experiments validate the theory and show that HAS is transferred even beyond the regimes covered by our analysis.

### 论文解读

#### 摘要翻译
CLIP通过图像和文本配对学习对齐，并未直接训练分类树。作者定义层级角分离（HAS）：一个概念的向量应比树外分支概念更接近自己的后代。实验发现图像嵌入在随机初始化时已有部分层级角结构，文本嵌入则主要在对比预训练后出现。固定图像特征、只学习文本特征的简化模型解释了这种不对称如何传递。

#### 方法动机分析
图像语义可能天然按子树聚集，但文本向量在训练前没有相同结构。问题是图文对比是否会把已有的图像层级传到文本，而非抹掉它。作者先量化每个祖先概念对后代与树外概念的余弦排序，再研究不同层级标签如何改变正样本集合。这一跨模态传递机制是论文要解决的理论挑战。HAS是描述性指标，不直接衡量识别准确率，也没有被加入CLIP训练损失。

#### 方法设计详解
对于树中每个非叶节点，作者计算它与最不相似后代的余弦相似度，并统计有多少树外节点超过这个阈值，得到HAS-Viol。经验分析采用ImageNet-1K图像与WordNet类别树，对比随机和预训练CLIP的图像、文本编码；同时做层级打乱空值以及图像样本分半，避免父节点和后代原型重复使用同一批实例导致假象。理论部分使用Unconstrained Features Model：固定图像嵌入，只优化每个概念的文本向量；图像标签可来自叶节点或其任意祖先。线性对比损失有闭式最优解，文本向量约为子树图像均值减去全局均值；补充的从头训练实验使用ImageNet-1K、CLIP ViT-B/32、批量1024训练32轮，AdamW学习率0.001，并比较路径一致、叶标签及打乱标签。作者再通过扰动范数与余弦间隔界推导转移条件。对InfoNCE则在强正则化下展开其最优解。

#### 方法对比分析
相比显式层级分类或双曲嵌入，本文不要求训练损失直接塑造树结构，而是研究普通对比对齐何时能传递已有层级。相比仅观察CLIP向量的相关性，HAS给出逐祖先的量化排序；相比非层级UFM分析，树状标签的共同出现会让祖先文本聚合后代图像。代价是理论假设较强：图像固定、概念覆盖均衡、原型均值一致；InfoNCE结果还依赖充分正则化。

#### 实验分析（精简版）
ImageNet/WordNet上，原始像素图像HAS-Viol为26.51%，随机ViT-L/14为27.88%，而打乱层级空值约51%与57%；预训练OpenAI CLIP图像嵌入降至2.95%。文本嵌入的层级结构则主要在训练后形成。分半测试排除了高斯特征因样本重用而出现的虚假低违反率。合成InfoNCE实验显示，违反率可在弱于定理要求的正则化下变为零；但真实CLIP图像原型不满足定理条件，作者只能报告经验转移和事后松弛量。打乱祖先标签后文本违反率超过一半。

#### 实用指南
研究表征几何时需报告树的构造、概念采样、祖先阈值及打乱空值。原型要用独立样本估计，并对照不同分类树和编码器；否则共享样本本身可能制造层级角度。若把HAS用于检索或分类，仍需另测任务性能。本文代码和数据来自ImageNet与WordNet设置；迁移到其他语言、类别体系或开放词汇前，应重新建立概念树并验证字幕是否沿祖先路径分布。

#### 总结
路径一致的图文配对会让祖先文本汇聚子树图像均值，因此可能继承图像已有的层级角结构。经验趋势跨模型较稳定，但理论保证依赖简化的固定图像模型及严格条件，真实CLIP尚未被定理覆盖。

1. 计算概念与后代及树外节点的余弦次序。
2. 用沿祖先路径配对的图像形成子树概念均值。
3. 通过对比对齐把图像几何传入文本表示。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17671v1)
- [arXiv](https://arxiv.org/abs/2611.17671v1)

---

<a id='2611.17629v1'></a>
## [Training-Free Acceleration of Multi-View 3D Transformers via Regime-Aware Attention Approximation](https://arxiv.org/abs/2611.17629v1)

**Authors:** Muhammed Izzet Sezer, Sevval Belkis Dikkaya

**Published:** 2026-11-13

**Categories:** cs.CV, cs.AI

**Abstract:**

Multi-view 3D transformers such as VGGT reconstruct a scene from a set of unposed images in a single forward pass, predicting camera poses, depth and 3D points, but the cost of their global attention grows quadratically with the num- ber of images. Accelerators borrowed from large language models assume that each token attends to only a few oth- ers; on these models they lose accuracy or gain little speed. We study the global attention of three multi-view transform- ers on seven datasets and find that every layer plays one of three roles: it averages over the whole scene, matches points across many images, or anchors on the first image, whose coordinates the outputs use. Building on this anal- ysis, we propose Regime-AWare ATTention Approximation (RAWATTA), a training-free method that runs attention in 8- bit precision and keeps 16-bit precision only in the layers where a worst-case calibration check requires it. Its fast set- ting, RAWATTA-Fast, also approximates attention in a few averaging and anchoring layers, while matching layers and the first image keep all their keys. On a 12 GB GPU at 50 im- ages, RAWATTA runs VGGT faster than the released code, mostly through our CUDA engine, and loses at most about one point of pose accuracy (AUC@3) on any dataset; RAWATTA-Fast is slightly faster and less accurate. Both fit 200 images, where the released code runs out of memory at 100.

### 论文解读

#### 摘要翻译
多视图三维Transformer能从无位姿图片同时恢复相机、深度与点云，但全局注意力计算随图片数量快速增长。通用稀疏化常损失几何精度。作者发现层间注意力有三种作用：汇总场景、跨视图匹配对应点、锚定首张参考图，并提出无需训练的RAWATTA，按层角色选择低精度与近似算子。

#### 方法动机分析
现有加速法往往希望每个token只关注少量其他token，或把相近token合并。可多视图几何的中间层要从许多图像收集匹配证据，错误丢弃会破坏点对应；输出坐标又依赖首帧，部分后层会大量关注它。核心挑战是哪些层能近似、用何种方式近似，同时保证最差评测域不超过精度预算。作者用实测注意力统计提出分层方案，并承认阈值需校准。

#### 方法设计详解
作者在VGGT-Ω等模型上统计每个全局层的归一化熵、首帧注意力质量等指标，标成三种regime。matching层保留稠密注意力；context层可对每帧单独注意或合并patch；anchoring层可合并非参考帧，但首帧全部token必须保留。合并键会加入log计数补偿。所有层使用INT8 QK和FP8 PV，校准若发现敏感层便把PV升至FP16。算法先跑校准样本，逐层搜索配置，再将整套设置在5个校准集上检查最差精度差。模型权重不变、不训练；计时使用作者编写的CUDA执行器。

#### 方法对比分析
相比通用稀疏注意力，RAWATTA不假设各层都稀疏，而是根据功能选择算子；相较固定层段压缩，作者逐层测量并为不同角色配置算子；相较单纯量化，还包含键值合并和首帧保护。消融显示在匹配层误合并会造成大幅精度下降，说明层位置关键。但CUDA引擎本身提供不少加速，论文把引擎和算法分别计时；部署者不能把全部加速都归功于regime近似。

#### 实验分析（精简版）
RTX 4070 SUPER、12GB、50图像时，RAWATTA比发布版快2.08倍，RAWATTA-Fast快2.23倍；相对发布版的最差数据集AUC@3分别下降1.03和1.14点。200图像时发布版在100帧已OOM；RAWATTA需12.51秒，Fast需10.56秒。Fast相对精确CUDA执行器在ETH3D 50帧下降1.65点，体现单集风险。实验跨7个数据集，但只有ETH3D与NRGBD在数据集层面完全未参与校准。对VGGT-1B和π3直接套用VGGT-Ω阈值会显著掉点，需重新搜索。

#### 实用指南
这项方法适合能在目标显卡和目标数据上独立校准的多视图重建团队。复现需固定分辨率、帧采样和基线引擎，分别报告方法加速与内核加速；检查最差数据集AUC、精度预算通过率、显存及长序列表现。论文未验证闭环SLAM或安全敏感部署，实际应用前应进行目标域校准和逐场景失效分析。

#### 总结
RAWATTA的关键不是把所有注意力变稀疏，而是按层的几何职责决定能否近似。首帧参考、跨图匹配和校准域覆盖是方法成败的核心边界；其显著收益依赖CUDA实现、显存预算和硬件。

1. 统计全局注意力并识别各层几何职责。
2. 保留匹配层，按角色压缩汇总层或锚定层。
3. 在目标域校准整套低精度与近似配置。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17629v1)
- [arXiv](https://arxiv.org/abs/2611.17629v1)

---

<a id='2611.17903v1'></a>
## [Zoom In, Scale Out: Metric Depth Models Mistake Digital Zoom for Moving Closer](https://arxiv.org/abs/2611.17903v1)

**Authors:** Rabeya Noor, Khandoker Nosiba Arifin

**Published:** 2026-11-13

**Categories:** cs.CV

**Abstract:**

Many new AI models estimate how far away things are, in metres, from a single photo. To do this they must also work out what camera took the photo. We check whether they really do with a simple test: digital zoom, that is, cropping a photo and enlarging the crop back to full size. Zooming moves nothing in the scene, so a correct model should give the same distance for every point still in the picture. The test is free, exact and needs no labels. Two numbers describe a model's reaction: its depth changes like $s^{-α}$ (ideal $α=0$) and its focal-length estimate like $s^β$ (ideal $β=1$), for zoom factor $s$. On 800 images from KITTI, NYUv2 and DIODE, the widely used Depth Anything V2 has $α=0.95$: it treats a zoomed photo almost as if the camera had moved closer, because it judges distance mainly from how large things look. Models that also predict the camera remove only part of this error ($α\approx0.31$-$0.36$), because their focal estimate grows too little. Given the true camera, Depth Pro and DA3-Nested keep their depth ($α\approx0$), so the error lies in their camera estimate. The reason: a zoomed photo is, pixel for pixel, a normal photo of a scene squashed in depth, so size alone cannot tell the two apart, and a focal estimator trained by regression stays close to the cameras it saw in training. Undoing this needs no labels or training and makes Depth Pro pass the test ($α=0.04$) at almost no cost on normal photos. Controls, including a real optical zoom lens, rule out blur and resizing artefacts. Finally, label-free fine-tuning for self-consistency lowers $α$ to 0.05 on KITTI and 0.07 on NYUv2 once a "blurry means zoomed" shortcut is removed; we report what this costs on normal and unseen photos.

### 论文解读
#### 摘要翻译
数字变焦可通过裁剪并放大照片实现，场景没有移动，留在画面中的物体深度也不变。作者用它检验单目度量深度模型是否正确处理相机焦距：Depth Anything V2几乎把变焦当成相机走近；焦距预测模型只部分缓解。给模型真实相机参数可定位误差来源，无标签修正则能改善一致性，但也带来精度代价。
#### 方法动机分析
单张图像中“近处广角”和“远处长焦”可能生成相同像素，因此米制深度必须正确估计或接收相机内参。传统测试集难以隔离相机因素。本文提出精确数字变焦：中心裁剪放大等价于焦距乘变焦倍数，物体真实位置不变。核心假设是，若模型理解相机变化，它应保持深度不变并同步放大焦距估计。
#### 方法设计详解
对同一图像施加1.25、1.5、2、3倍裁剪放大，比较相同可见像素在原图与变焦图上的预测。以幂律拟合深度变化s^-α和焦距变化s^β，理想值分别为α=0、β=1；α+β≈1是模型依赖表观物体尺寸的几何特征。评估覆盖纯像素、预测焦距和显式相机输入三类模型，并设置模糊、真实光学变焦、JPEG、偏心裁剪和真实焦距等对照。作者还提出FocalUnshrink反转焦距预测的收缩，以及EFT+B无标签一致性微调；后者增加模糊不变项，避免网络把“模糊”误作“变焦”。
#### 方法对比分析
相较常规深度准确率榜单，变焦审计不需要新标签，能直接检验相机等变性；相较仅检查裁剪鲁棒性，它给出焦距更新后的明确几何基准。核心创新是把相机—深度歧义变成可复现的无标签测试，并用α、β定位偏差。它适用于报告米制深度的模型，不能单独说明普通图像的绝对准确率。
#### 实验分析（精简版）
800张KITTI、NYUv2及DIODE图像上，DA-V2-S的α为0.95，3倍变焦时预测距离大幅缩短；Depth Pro为0.31，给它真实焦距后降至-0.03。模糊对照影响仅1.3%，真实光学和数字变焦对DA-V2-S的α均为0.76。四个数据分组各抽200张图，置信区间按场景聚类重采样；3倍变焦时该模型的AbsRel由未变焦0.317升至0.652。作者还发现FocalUnshrink可使Depth Pro α降至0.04，且1×尺度误差基本持平。EFT+B把KITTI/NYUv2的α降至0.05/0.07，但KITTI未变焦AbsRel由0.138升至0.175，DIODE域外效果也较弱。
#### 实用指南
测试只需原图、裁剪放大和模型输出；若部署会数字变焦或裁图，应同步更新内参。FocalUnshrink不需训练，但需目标域无标签图像；EFT+B需约3000步微调并要监控模糊捷径及域外精度。代码公开，训练设置包含392像素输入、单T4 GPU和小批量。
#### 总结
核心思想：变焦不应改变场景深度
1. 对图像裁剪并放大，精确更新焦距。
2. 比较相同像素的深度与焦距响应。
3. 用控制实验区分相机误差和图像退化。
4. 修正后重新检查绝对精度与域外表现。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17903v1)
- [arXiv](https://arxiv.org/abs/2611.17903v1)

---

<a id='2611.17658v1'></a>
## [Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost](https://arxiv.org/abs/2611.17658v1)

**Authors:** Christopher Malili

**Published:** 2026-11-13

**Categories:** cs.RO, cs.AI, cs.LG

**Abstract:**

Imitation policies for robot manipulation are often trained on noisy object positions, and they learn to discount the noisy reading. We show how much they discount it, and why. Because demonstrators steer toward the object, the robot's own state carries information about where the object is. A trained policy combines this proprioceptive anchor with the reading and weights the reading by $α= V/(V + s^2)$, where $s^2$ is the reading noise and $V$ the anchor's error variance. $V$ belongs to the learner and is far smaller than a linear model of the demonstrations implies, which is why classical errors-in-variables (EIV) regression overestimates $α$. Measured once, $V$ predicted ten held-out sensitivities across two robomimic tasks, two learners and four new noise levels, with mean absolute error 0.025 against 0.37 for classical EIV, and a parameter-free bias test rules out the simpler explanation that the policies just overestimate the noise. The discount does not carry over to deployment, where the policy produces its own state: closed-loop sensitivity to a persistent reading offset exceeded $α$ in all 15 policies that passed the registered gates, although the stricter registered prediction, that all would follow the offset fully, held for 8 of them. Post hoc, lower $α$ went with lower success, steeply for diffusion, and a registered prospective test confirmed this for diffusion on Can at two new noise levels: the measured $α$ predicted success before any rollout, within 0.03 (95% intervals $\pm 0.08$), although training noise predicts it equally well. Every test was registered before its data, including the classical benchmark that failed, and all other analyses are labelled post hoc; all results are in simulation with synthetic Gaussian noise.

### 论文解读
#### 摘要翻译
模仿策略在训练中见过带噪的物体位置后，会学会降低对读数的依赖。作者发现，机器人本体状态也能提示物体位置：示范者朝物体运动，因而机械臂姿态形成“本体感觉锚”。策略对带噪读数的权重遵循α=V/(V+s²)，但V是学习器自身的锚点误差，远小于传统回归从演示估计的方差。开环预测在新噪声水平上准确；闭环持续偏差却会影响机器人自己生成的状态，保护因此消失。
#### 方法动机分析
经典errors-in-variables回归把输入读数的噪声与演示数据的真实位置方差结合，预测回归权重衰减。深度模仿策略还可从干净的机器人状态估计目标位置，所以它并非只依赖噪声读数。论文假设策略会按两个线索的相对可靠度加权；其中锚点方差V属于学习器，需从一个校准策略测量。部署时状态由策略自身动作产生，持续读数偏置可能被误认为真实物体位移。
#### 方法设计详解
输入→输出流程是：状态与带噪位置进入策略，先由本体状态形成锚，再用方差比融合感知读数，输出末端动作。作者在robomimic的Square螺母装配和Can拾罐仿真任务中，对物体位置注入独立高斯噪声。比较四层512单元MLP与扩散策略，各用200条人工演示、学习率1e-4、批量256训练40000步；扩散预测16步动作块，每次执行8步。通过在抓取前对感知位置做±5毫米有限差分，测得噪声策略相对干净策略的平移动作敏感度α。先在12毫米噪声策略上估计V，再按α=V/(V+s²)预测其他噪声级，测试规则事先登记。另以清洁读数下的偏差区分“学到本体锚”与“简单把噪声估大”，并在闭环中施加持续读数偏移，观察末端瞄准位置和成功率。
#### 方法对比分析
与传统EIV相比，公式形式相似，但V来自策略本身的条件锚点，而不是线性拟合演示的误差；所以它能区分不同学习器。论文还把开环感知折扣与闭环抵抗偏置分开测量，揭示前者不能直接代表部署鲁棒性。适用范围是本体状态较低维、目标位置受控噪声影响的模仿任务，尚未覆盖视觉端到端感知。
#### 实验分析（精简版）
Square和Can两任务、两类策略的10个预注册新噪声级预测，锚定律平均绝对误差0.025且10/10在容差内；传统EIV误差0.37且0/10通过。闭环可判别的15个策略中，8个完全跟随偏置、7个部分跟随，没有一个保留开环保护。Can扩散策略两个新噪声级的成功率预测0.760/0.402，实测0.758/0.375。所有实验均为合成噪声仿真，且α与训练噪声仍混杂。
#### 实用指南
复现需保持演示、噪声种子与干净策略配对，按预注册门槛筛除动作饱和或失败策略。部署评估应同时测开环灵敏度、持久偏移、真实物体位移与闭环成功；不应把低α直接当作鲁棒性。机器人实测和结构化感知误差仍待验证。
#### 总结
核心思想：状态线索会成为噪声感知锚点
1. 示范轨迹使机器人状态携带目标位置信息。
2. 策略按锚点与读数方差融合线索。
3. 校准一个噪声级，预测其他开环敏感度。
4. 在闭环加入持续偏置，检查保护是否失效。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.17658v1)
- [arXiv](https://arxiv.org/abs/2611.17658v1)

---

