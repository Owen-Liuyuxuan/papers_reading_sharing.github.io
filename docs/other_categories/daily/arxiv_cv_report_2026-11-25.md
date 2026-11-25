time: 20261125

# Arxiv Computer Vision Papers - 2026-11-25

## Executive Summary

## 2026-11-25 计算机视觉与机器学习论文执行摘要

本期 10 篇论文中，机器人学习与世界模型占据主线：研究者正把视频或轨迹中的动力学压缩成可用于动作选择、长时控制和规划的表示，同时补齐仿真环境、训练数据配比与部署效率等基础设施。另有工作探索统一的 4D 感知与生成空间、示例驱动的开放类别分割，以及带可验证奖励的多模态工具使用。

**值得关注的进展**包括《World Models as Native Latent Action Models》：它从预训练视频世界模型中读出潜动作，并在多个本体的视频上训练策略；仿真和有限实机结果显示出潜动作接口的潜力，但跨场景迁移证据仍有限。《ABI》提供另一条路线：不更新冻结策略，而在测试时用无奖励轨迹学到的表示筛选候选动作块；多个任务上的提升值得关注，效果则依赖候选动作落在数据支持范围内及排序条件成立。《RoboUniverse-10K》把真实操作视频转成大规模可交互仿真任务，形成机器人后训练与评测资源；其留出集微调成绩有提升，但双臂任务表现较弱，真实机器人证据也有限。

效率和长时序是另一条交叉趋势。《Motion Mamba 2.0》将递归状态、持续内容记忆与局部注意力结合，并报告了窗口生成加速及 Unitree G1 系统任务结果；训练历史较短，长程能力依靠自回归展开。《PTQ4WAM》把量化精度和目标 GPU 执行路径联合选择，在多种工作负载上降低延迟与内存，但任务分数并非处处保持，且硬件验证限于 RTX 4090。《ProMix》用小型代理模型为机器人模仿学习优化数据域配比，报告平均及最差域成功率提升；成本比较未涵盖全部调参与训练开销，仍需跨语料验证。

视觉与规划方面，《WorldLatent》尝试在统一 4D 潜空间中处理视频几何感知、生成和编辑，显示共享表示的潜力；其真实数据深度与相机标签大量来自伪标注，生成评测规模有限。《SAYD》以查询图像生成类别无关掩码、再由示例原型分类，并显式支持未知类别拒识，体现了“示例定义类别、查询决定边界”的设计思路；目前证据来自静态图像基准。《MOVER》将历史扰动得到的事件路由信号用于规划世界模型，nuScenes 指标有改善，但碰撞差异未显著，不能把事件摘要解释为因果注意。《Visual-RFT++》展示了可验证奖励对视觉任务和多模态工具使用的训练价值；小测试集、合成畸变数据及外部检索依赖限制了部署层面的结论。

**新兴方向**是让模型在推理或执行阶段更有效地使用已有能力：潜动作接口、测试时动作筛选、历史记忆压缩、任务导向量化和事件路由都在减少对端到端重训或昂贵推理的依赖。与此并行，真实到仿真的任务构建和数据混合优化正成为机器人规模化训练的重要支撑。后续值得检验的共性问题包括跨场景与跨本体泛化、复杂物理和双臂任务覆盖、仿真到实机迁移，以及效率收益在不同设备上的可复现性。

**优先阅读全文：**若关注机器人动作表征与策略，先读《World Models as Native Latent Action Models》和《ABI》；若关注评测与训练基础设施，读《RoboUniverse-10K》及《ProMix》；若关注长时控制和部署成本，读《Motion Mamba 2.0》与《PTQ4WAM》。对通用视觉表示感兴趣的读者可优先看《WorldLatent》和《SAYD》，并结合其伪标注、基准范围及静态图像证据边界评估结果。

---

## Table of Contents

1. [World Models as Native Latent Action Models](#2611.30898v1)
2. [ABI: Actor-Free Behavior-Induced Representations for Zero-Shot Test-Time Guidance](#2611.30636v1)
3. [RoboUniverse-10K: Toward the Scaling Era of Agentic Robotics via Real-to-Sim Environments](#2611.31002v1)
4. [Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control](#2611.29897v1)
5. [ProMix: Efficient Data Mixture Optimization for Robotic Imitation via Proxy-Reference DRO](#2611.29873v1)
6. [MOVER: Planning Beyond Motion](#2611.30022v1)
7. [WorldLatent: A Unified 4D Latent Space for World Perception and Generation](#2611.31209v1)
8. [Segment Anything In-Context via Visual Demonstrations](#2611.29896v1)
9. [Visual-RFT++: Extending Visual Reinforcement Fine-Tuning to Multimodal Tool Use](#2611.30193v1)
10. [Quantization for Action, Execution for Deployment: Joint Precision and Execution Selection for World Action Models](#2611.30195v1)

---

## Papers

<a id='2611.30898v1'></a>
## [World Models as Native Latent Action Models](https://arxiv.org/abs/2611.30898v1)

**Authors:** Qinyue Li, Weijie Qiao, Qingyi Tao, Zerui Gong, Zhe Yang, Quanzhou Li, Taolin Zhang, Jinglin He, Zhonghua Wu, Chen Change Loy

**Published:** 2026-11-23

**Categories:** cs.CV, cs.LG, cs.RO

**Abstract:**

Large and diverse video sources, such as egocentric human video and heterogeneous robot recordings, carry rich dynamics but come either without actions or without a shared action space. Latent action models (LAMs) can recover these dynamics, but they train their inverse- and forward-dynamics models (IDM, FDM) from scratch, trading off pixel reconstruction against latent capacity and accuracy. Pretrained video world models natively generate high-fidelity dynamics, yet represent them densely, entangled across all layers. Viewing the world model as a LAM, we show that these dynamics can be extracted with minimal changes to the model. Native LAM operates one pretrained video diffusion transformer as both IDM and FDM, with learnable dynamics registers as the only causal bridge between the two, and trains on counterfactual temporal pairs that differ in motion alone, so that the registers distill a video's dynamics into a compact native latent action. We further propose a pretraining paradigm for vision-language-action (VLA) policies, Pretraining Actions from World models (PAWlicy), in which Native LAM serves as a dynamics teacher on large-scale video; unlike world action models, it never generates video. Native latent actions outperform existing LAMs on the LARY benchmark in both action classification and control regression, and replay the same motion on new robots, in new scenes, and from human hands to robots, although no transfer objective is used in training. With a 3.8B model, PAWlicy reaches a 93.6% average success rate on RoboTwin 2.0 trained with domain-randomized demonstrations, the highest among VLAs. On RoboDojo, it is competitive on every dimension we evaluate, and its Precision, evaluated by us, is on par with the leading leaderboard entries. On real robots, it improves over the pi_0.5 baseline on both tasks. Both stages use only public data and a publicly released world model.

### 论文解读

#### 摘要翻译
人类第一视角视频和不同机器人录像包含丰富运动规律，却常无动作标签或统一动作空间。常规潜动作模型从头学习逆、正向动力学，需折衷图像重建与潜变量能力。Native LAM 复用预训练视频扩散 Transformer 读取并重放潜动作，以反事实视频对逼出运动信息；PAWlicy 再用该监督预训练策略，不生成未来视频。论文在 LARY、仿真和实机上评估。

#### 方法动机分析
人类与异构机器人动作难对齐，而无标签视频更丰富。既有 LAM 从头训练编码器、解码器，易泄漏未来画面、丢失动作或学到场景变化。作者假设预训练世界模型已有动力学知识，只需暴露紧凑动作接口，并迫使生成器依赖它。

#### 方法设计详解
Native LAM 将同一 Kairos 3.1 视频扩散 Transformer 运行在两种模式。首先把干净视频编码为视频 token，并附加八组、每组四个可学习动力学寄存器。逆动力学模式用只读注意力让寄存器读取整段视频，再由输出 MLP 得到 32 个维度为 128 的潜动作 token。正向动力学模式仅输入首帧、潜动作和带噪未来，不给文本或干净未来；输入 MLP 将动作 token 映射到主干宽度，模型按 flow matching 从噪声生成后续视频潜变量。

训练时从视频构造首帧相同而运动不同的反事实对，包括暂停、变速、静止、重定时或反向。每对共享上下文、噪声和扩散时间，因此生成差异只能由潜动作解释，降低模型忽略潜变量的后验坍塌。仅 AgiBot World 与 DROID 的标签用于末端执行器运动辅助监督，其余视频不需动作或文本标签。之后冻结该模型作为教师，为视频缓存潜动作。PAWlicy 使用视觉语言模型、潜动作 DiT 和动作 DiT：输入当前图像、指令及可随机丢弃的过去帧，预测一个过去潜动作组和三个未来组；动作专家通过共享注意力依据潜动作生成动作块。策略预训练为 50k 次迭代、32 张 GPU、全局批量 512；推理时两个 DiT 同步做 10 步 Euler flow-matching。

#### 方法对比分析
从头训练的 LAM 需要独立编码器、解码器；世界动作模型则联合预测动作和高维未来视频。创新点是让预训练世界模型自身兼任两者，仅以寄存器传递潜动作。PAWlicy仅对齐紧凑运动表征，避免策略阶段生成视频；但效果仍依赖教师动力学先验及反事实构造。

#### 实验分析（精简版）
LARY 上 Native LAM 的动作分类均值为 71.77%，高于先前最佳 WLA3 的 70.75%；控制回归 MSE 为 0.25，优于 0.37。使用随机化演示时，RoboTwin 2.0 平均成功率为 93.6%，在两项 AgileX NERO 实机任务上，堆叠方块成功率 76.7%、鞋子入盒 90.0%，高于匹配输入设定的 π0.5。跨机器人与场景重放主要是定性结果，初始几何基本保持；真实任务仅两项、各 30 次，RoboDojo 未评估需要记忆模块的 Memory 维度。

#### 实用指南
两阶段均用公开数据和 Kairos 3.1 权重。复现需混合机器人、人类与仿真视频，构造时间变换反事实对；运动学标签仅用于两个来源。Native LAM 训练 13k 次、32 张 GPU；PAWlicy 训练 50k 次、32 张 GPU、批量 512。标签可缓存，策略推理无需视频生成主干。换用其他视频 DiT 或大幅改变几何尚待验证。

#### 总结
**把世界模型的运动知识压成动作接口**
1. 从预训练视频模型附加动力学寄存器，读出紧凑潜动作。
2. 用首帧相同、运动不同的视频对，逼迫生成器依赖潜动作。
3. 冻结教师，为人类、机器人和仿真视频缓存潜动作标签。
4. 用潜动作预训练策略，再以目标机器人演示微调并闭环执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.30898v1)
- [arXiv](https://arxiv.org/abs/2611.30898v1)

---

<a id='2611.30636v1'></a>
## [ABI: Actor-Free Behavior-Induced Representations for Zero-Shot Test-Time Guidance](https://arxiv.org/abs/2611.30636v1)

**Authors:** Yifei Yao, Yude Zou, Guanbo Jiao, Jianming Ma, Xing Gao, Chunhua Shen

**Published:** 2026-11-23

**Categories:** cs.RO, cs.LG

**Abstract:**

Behavior-cloning policies show strong ability in sequential control by generating coherent multi-step action chunks, yet their purely imitative objective limits zero-shot generalization to unseen tasks. While offline reinforcement learning offers principled policy improvement, existing methods either require task-specific reward supervision or remain tied to a specific actor during critic training or policy deployment, inflating training cost and restricting cross-policy generalization. We observe that evaluating actions under the data-collecting behavior largely preserves their relative value ordering across target policies, offering a way around actor coupling. Combining this property with reward-free successor prediction, we propose Actor-Free Behavior-Induced Representations (ABI) to learn multi-horizon action-chunk representations from unlabeled trajectories without reward supervision or actor training. At test time, pairing the frozen representation with lightweight task readouts forms a chunk critic that guides target policies without parameter updates and transfers directly across distinct unseen policies. Our analysis establishes when the value ordering is preserved across policies and when selecting among proposed chunks improves closed-loop return. Across 65 locomotion, navigation, and manipulation tasks, ABI's chunk critic guides weak imitation policies to match or surpass zero-shot RL baselines with high sample efficiency, outperforms existing zero-shot critics on identical proposals, and improves externally trained RL actors without retraining. From raw pixels, ABI delivers substantial guidance gains and surpasses prior zero-shot RL baselines on visual manipulation, highlighting its potential for visuomotor control.

### 论文解读

#### 摘要翻译

行为克隆策略能生成连贯的多步动作，但单纯模仿难以应对未见任务。ABI 从无奖励轨迹学习动作块的后续状态表示，再以轻量任务读出形成测试时 critic，在不更新策略参数的情况下筛选或引导候选动作。作者分析何时跨策略动作排序能够保留，并在运动、导航和操作任务中评估效果。

#### 方法动机分析

常见离线强化学习依赖任务奖励或与特定 actor 联合训练，限制了新任务和新策略的复用。ABI 的关键假设是：在数据收集行为下评估动作块，虽然不等于目标策略的真实价值，但候选间排序误差可能较小。误差主要来自两部分：critic 对行为价值估计的误差，以及动作块末端按数据行为或目标策略继续时的价值差异。后一项在候选端点间变化越小，排序越可能迁移。

#### 方法设计详解

输入是轨迹中的状态（或像素）、长度为 H 的动作块、块内状态到达序列和下一个记录动作块。编码器得到当前状态与到达状态表示，预测器估计折扣后的后续表示；训练目标把块内折扣特征与 γ^H 倍的 EMA 目标预测相加。H 从 {4,8,16,32} 采样，预训练不使用奖励、任务输入或 actor。状态输入采用 16 步历史窗口；像素输入使用三帧 64×64 RGB。

训练后冻结主干。测试时从最多 10,000 个缓存状态构造任务读出：密集奖励用特征矩线性回归，稀疏目标用类别密度读出。冻结 proposer 生成动作块，critic 可用 best-of-N 选分数最高者，或用低频动作子空间中的价值梯度引导生成；执行可整块执行，也可只执行首步并重规划。排序界给出直觉：共同的状态分数偏移会抵消，候选的误差离散度和端点续接差异才会改变名次。滚动执行还要扣除未执行尾部的相对价值，不能把整块分数直接当作首步收益。

#### 方法对比分析

本质区别是：TD-JEPA 等方法用联合训练的 actor 产生 bootstrap，ABI 则用数据里的下一记录动作块来学习 actor 无关的表示；one-step FB 评估单步动作，ABI 直接评估多步动作块。创新点在于把行为续接的价值误差拆成候选间排序误差和端点续接差异，从而解释何时可跨策略复用。ABI 不是对任意策略都精确的 Q 函数：当新策略与数据行为在块末端产生明显不同的后续价值时，排序保证会变弱。数据覆盖宽时可尝试梯度引导，数据窄时优先在策略候选内排序。

#### 实验分析（精简版）

作者在 65 项任务上测试，包含四个 DMC 域和九个 OGBench 数据集，并比较三个训练种子。弱 Flow BC 经 ABI 引导后，DMC 平均回报由 58.4 升至 655.2，OGBench 成功率由 3.0% 升至 38.6%。在相同候选上比较 critic，ABI 的 OGBench 成功率为 38.6%，TD-JEPA 为 16.7%。原始像素输入下视觉 OGBench 也由 5.3% 提升到 32.4%。但像素结果仍弱于状态输入，且窄数据上大幅梯度引导可能离开数据支持范围。

#### 实用指南

复现设定包括每个数据集 62,500 次预训练更新、batch size 2,048，AdamW 学习率从 3×10⁻⁴ 余弦衰减至 3×10⁻⁵；状态实验用三种训练种子。推理任务读出从缓存状态获取奖励或目标标签，策略仍冻结。论文列出项目页，但正文未说明代码发布情况。迁移到机器人时，应先检验轨迹是否覆盖 proposer 会提出的动作块，并比较数据行为与目标策略在块末状态的续接差异；对窄覆盖数据，候选筛选比大幅梯度推送更稳妥。

#### 总结

核心思想：离线学评估，测试时选动作。

1. 收集无奖励轨迹，记录状态和动作块。
2. 训练多视野后续状态预测器，并冻结表示。
3. 按新任务读取价值，对冻结策略候选排序或引导。
4. 在覆盖与续接条件成立时执行并逐步重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.30636v1)
- [arXiv](https://arxiv.org/abs/2611.30636v1)

---

<a id='2611.31002v1'></a>
## [RoboUniverse-10K: Toward the Scaling Era of Agentic Robotics via Real-to-Sim Environments](https://arxiv.org/abs/2611.31002v1)

**Authors:** Jinkun Liu, Yingze Wang, Situ Wang, Deyao Hong, Mingju Gao, Wenyi Li, Yizhe Chi, Youjie Zheng, Kaisen Yang, Xiangyang Wang, Bingxiang He, Yuxin Zuo, Boyuan Chen, Yue Chen, Cheng Qian, Tianxing Chen, Huichi Zhou, Huaxiu Yao, Hao Tang, Hao Zhou, Ya-Qin Zhang, Calvin Xiao, Tiga Altman, Huan-ang Gao

**Published:** 2026-11-23

**Categories:** cs.RO, cs.CL

**Abstract:**

Agentic Robotics, in which foundation models act as coding agents to control robots through executable programs, has emerged as a compelling paradigm for embodied intelligence. Yet unlike software engineering, where massive executable environments power post-training, robotics remains constrained by small, manual benchmarks and in-distribution evaluation. To bridge this gap, we introduce RoboUniverse-10K, 10,807 interactive, resettable, and verifiable physics environments built from real-world videos via an automated Real2Sim and quality assurance pipeline. Each runs in an isolated, Harbor-compatible harness that hides privileged states and grading logic to mirror real-world sensing. From the same pipeline, we hold out 500 unseen tasks to establish RoboUniverse-Verified, an expert-verified zero-shot benchmark. Evaluating 11 closed- and open-source models reveals a huge gap: frontier models reach 85.8\% success, while open-source baselines lag at 2.6\% to 10.0\%, bottlenecked by low-level grasping, deformable-object manipulation, and post-contact physical control. Finally, we collect about 17K multi-turn trajectories totaling 0.79B tokens with GPT-6 Astra and perform full-parameter supervised fine-tuning on Qwen3.6-35B-A3B, yielding RoboUniverse-35B-A3B. Our model achieves substantial zero-shot gains on RoboUniverse-Verified, lifting the success rate from 0.8\% to 21.2\%, while achieving zero-shot task completions across LIBERO-MAX, RoboCasa, RoboTwin 2.0, and a real Franka arm. We open-source the RL-ready environments and model to lay the foundation for large-scale reinforcement learning in Agentic Robotics.

### 论文解读
#### 摘要翻译
RoboUniverse-10K尝试解决机器人智能体训练环境规模小、测试任务与训练任务相似的问题。作者从真实机器人操作视频自动重建仿真任务，发布10,807个可交互、可复位、可自动判分的环境，并留出500个专家核验任务测试零样本能力。11个模型的比较显示，顶尖闭源模型成功率显著高于开放模型。研究还用约17K条教师轨迹微调开放模型，在留出任务和若干外部平台上观察到提升。

#### 方法动机分析
机器人任务难像软件问题那样直接从仓库批量抽取，因为每项都要构造物理场景、控制接口和成功判据。手工基准规模有限，训练和测试若环境相似也难证明泛化。作者的假设是，真实演示视频可提供任务目标和场景依据，代码智能体配合感知工具及物理仿真，可以自动生成大量可执行环境；独立验证则能降低自动生成任务中的错误判分。

#### 方法设计详解
流程输入真实操作视频，先筛选完整演示并去除近似重复任务，再抽取物体、初态、目标，写成自然语言任务说明。构建智能体借助SAM 3.1分割、DA3估深、资产库和Isaac Sim搭建机器人与物体场景，编写参考控制程序和任务verifier。通过渲染对照和物理执行迭代修正碰撞、质量、摩擦和关节驱动。独立QA智能体检查仿真是否保留任务语义、重置后控制是否可复现，以及verifier能否拒绝穿透或瞬时成功并接受合理替代解。通过后打包为隔离的Harbor任务。被测模型只见图像和关节状态，通过关节目标控制机器人；它无法读取物体真值、参考程序、验证器或中途分数。每项任务只允许一次尝试。训练阶段用GPT-6 Astra生成约17K条、0.79B tokens轨迹，对Qwen3.6-35B-A3B做全参数语言骨干微调，冻结视觉编码器与对齐器，使用全局batch 64、最长上下文256k，约消耗1,500 H100等效GPU小时。

#### 方法对比分析
相较于通常由人工编写、规模为几十至数百项的模拟基准，本作把视频重建、参考控制和自动判分整合成统一生产线，并提供独立留出任务。相较只调用预制高层技能的具身测试，这一接口要求模型自行将视觉观察转成代码和关节动作，考查低层闭环控制。其关键差异是把运行环境与评分逻辑隔离，同时要求第三方QA审查场景、复位及判分。

#### 实验分析（精简版）
RoboUniverse-Verified包含500项任务，200项单臂、300项双臂，覆盖八类；10位专家审查任务目标、仿真可行性和验证逻辑。GPT-6.2 Astra在该集成功率为85.8%，开放模型中最高的DeepSeek V4.1 Flash为10.0%。SFT将Qwen3.6-35B-A3B从0.8%提高到21.2%，其中单臂由2.0%升至40.5%，双臂仅升至8.3%。外部基准有零样本成功案例，真实Franka展示一个绘图任务；这些结果未提供广泛外部任务成功率。作者也指出场景动态仍较简化，长时序任务不足。

#### 实用指南
适合研究机器人代码智能体、仿真任务生成及可验证后训练的读者。复现时需关注数据源和任务去重规则、隔离Harness、QA覆盖以及模型所用控制接口；基准采用每任务一次尝试和独立终态判分。论文报告了训练关键超参数与GPU时数，并计划开放环境和模型。可迁移的思路是用真实视频引导可执行环境构建，并将观察、动作和判分边界清晰分开；迁移到其他机器人或自动驾驶领域仍需重新建模并实证验证。

#### 总结
核心思想：让真实演示规模化生成可验证机器人任务。
1. 从多源真实操作视频筛选并规范任务。
2. 智能体在仿真中建场景、控制器和判分器。
3. 独立QA审查保真、复位重放与验证逻辑。
4. 用隔离环境训练，再以留出任务测泛化。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.31002v1)
- [arXiv](https://arxiv.org/abs/2611.31002v1)

---

<a id='2611.29897v1'></a>
## [Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control](https://arxiv.org/abs/2611.29897v1)

**Authors:** Yue Huang, Tianshan Zhang, Zeyu Zhang, Jayeon Kang, Akide Liu, Sungjoon Choi, Dacheng Tao, Hao Tang

**Published:** 2026-11-23

**Categories:** cs.RO, cs.CV

**Abstract:**

Long-horizon, real-time controllable motion generation is essential for interactive animation and humanoid robotics, where an agent must respond to changing instructions and spatial goals while moving. Existing autoregressive diffusion methods generate motion incrementally, but bounded attention caches limit access to distant history, while repeated denoising introduces substantial inference overhead. We present Motion Mamba 2.0, a Mamba-based root-body diffusion architecture that integrates local attention, resumable recurrent states, and persistent content memory (PCM) for long-horizon controllable motion generation. PCM transforms compressed motion history from Mamba recurrent states, together with state changes and text-conditioned features, into dynamically updated content slots accessible to local attention. To enable efficient inference, we further develop an execution scheme tailored to this architecture, combining CUDA Graph replay, fused kernels, and execution-flow simplification while preserving the canonical history update. Evaluations on retargeted HiPHI demonstrate accurate terminal control over rollouts of up to 39.52 s, while short-horizon comparisons on BONES Seed and retargeted HumanML3D show that the model also retains strong spatial control and text-motion matching. On an RTX 4090, the optimized execution reduces median complete-window latency from 1685.35 to 276.14 ms, achieving a median paired speedup of 6.138$\times$ while preserving identical outputs and committed states in the tested rollouts. Moreover, we conduct extensive real-world evaluation on the Unitree G1, demonstrating sequential text-guided action composition, expressive motion, and interactive locomotion through keyboard and mouse control.

### 论文解读

#### 摘要翻译
Motion Mamba 2.0面向长时域、实时的人形运动控制。它把局部注意力、Mamba递归状态和持续内容记忆（PCM）结合，使压缩历史可被注意力读取；另以CUDA Graph等手段降低逐窗扩散的推理开销。模型在最长39.52秒的评测和Unitree G1实机任务中验证了控制能力。

#### 方法动机分析
逐窗生成便于执行中接受新指令，但注意力缓存会丢弃旧特征，无界扩展又增加成本；Mamba能压缩历史，却不能让注意力直接选择读取。作者假设把已接受状态、状态变化与文本写入固定槽位，可补充近期缓存。去噪候选须共享未变历史，避免采样步骤被误认为动作时间。

#### 方法设计详解
输入为文本、空间目标、扩散时间和已接受运动。每窗生成52帧（25 Hz下2.08秒）：root扩散器先预测全局根运动，再转为局部条件供body扩散器生成关节动作；每个去噪步都依次更新root和body。两套网络的混合层并行运行局部注意力与可恢复Mamba-3，学习门融合两路结果。注意力读取最近四窗缓存和PCM槽位，Mamba携带固定大小递归状态。

PCM逐层将接受后的Mamba状态投影成摘要，并计算相邻摘要差；摘要、文本调制差分与文本特征构成候选槽内容，经受限门控写入固定槽。去噪过程只读槽位。生成窗口被接受后，模型才统一提交注意力缓存、Mamba状态和PCM。推理端跳过无用状态物化、融合Mamba转移预处理，并以CUDA Graph回放root/body的三路引导分支；推理用RTX 4090、batch=1、10步DDIM。

#### 方法对比分析
PCM通过状态摘要、差分和文本条件建立了递归状态到注意力内容的映射，补充滑动注意力的压缩历史。其接受后更新规则还把历史提交与多步采样解耦；这不同于仅延长缓存或只减少采样步数。它不保证找回任意远端动作。相比只保留循环状态，PCM显式增加内容写入和注意力检索路径，适合约束和指令持续变化的任务。

#### 实验分析（精简版）
HiPHI最长39.52秒时，根/末端误差为7.55/13.64 cm，基线对应最佳值为7.78/14.84 cm。完整窗口p50从1685.35降至276.14 ms，成对中位加速6.138倍。G1四类实机任务100次中94次完整成功，未观察到跌倒。训练最长历史为7.84秒，长程测试依赖自回归展开；实机结果来自远端生成与SONIC跟踪的整套系统，尚未验证跨平台泛化或物体操作。

#### 实用指南
代码仓库与项目网站已公开，包含模型实现与项目演示。训练用BONES Seed、HiPHI和HumanML3D统一到G1表征，batch 512、更新100,000次，历史最长196帧。复现或迁移时需另测通信、长程误差、目标切换和闭环安全；训练设定未直接覆盖39.52秒生成历史。

#### 总结
核心思想：递归历史写入可读记忆

1. 接收指令、空间目标和已接受运动。
2. root-body扩散逐窗生成全身动作。
3. Mamba总结历史，PCM按文本更新记忆槽。
4. 接受窗口后提交状态，并以CUDA Graph加速。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.29897v1)
- [arXiv](https://arxiv.org/abs/2611.29897v1)

---

<a id='2611.29873v1'></a>
## [ProMix: Efficient Data Mixture Optimization for Robotic Imitation via Proxy-Reference DRO](https://arxiv.org/abs/2611.29873v1)

**Authors:** Qie Sima, Wei Xue, Yike Guo

**Published:** 2026-11-23

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models excel in open-world manipulation, yet their performance often saturates when training data are naively pooled across heterogeneous robots, tasks, and environments. Distributionally robust data mixture optimization, exemplified by Re-Mix, can recover much of this performance, but its weight search requires training full-scale policies twice on the entire corpus, placing it out of reach for most laboratories. We present ProMix, a compute-efficient instantiation of the proxy-reference DRO recipe for robotic imitation learning: a 15M-parameter proxy learns per-domain sampling weights by minimizing worst-case excess loss against a frozen reference model, and the learned weights transfer, without re-tuning, to VLA backbones up to 7B parameters. We further provide, to our knowledge, the first controlled analysis of the reference model's role in robotic DRO, showing that removing it induces over-pessimism toward noisy, low-quality domains (-3.8% average and -14.3% worst-domain success rate), and that entropy regularization is needed to prevent weight collapse. Evaluated on 9500 real-world episodes from the Open-X Embodiment dataset, ProMix improves the average success rate over uniform mixing from 68.4% to 76.2%, matching Re-Mix within evaluation noise while reducing mixture-optimization cost to a backbone-independent 11 GPU-hours - versus 16 GPU-hours at the 50M comparison scale and 192 GPU-hours at 7B for full-scale searches - with a +17.1% absolute gain on the worst-case domain.

### 论文解读

#### 摘要翻译
机器人视觉语言动作模型需要汇集不同机器人、任务和环境中的示范，但直接混合常使性能停滞。已有分布鲁棒混合方法 Re-Mix 能改善效果，却需在全量数据上以最终策略规模训练参考模型和鲁棒模型，搜索成本高。ProMix 让 1500 万参数的代理模型相对冻结参考模型学习域权重，再将权重直接迁移到最多 70 亿参数的模型。作者还分析了参考模型和熵正则的作用，并在 9500 条 Open-X 真实机器人回合上报告了对均匀混合的提升。

#### 方法动机分析
数据域之间既有难度差异，也有示范噪声差异。普通最坏域优化会把高权重给绝对损失最大的域，即使它的误差主要来自不可约噪声。ProMix 的核心假设是，小模型可以发现哪些域相对一个均匀训练的参考模型仍有可学习的差距，而这类相对难度足以指导更大的策略训练。

#### 方法设计详解
输入是按机器人或任务划分的示范域。第一阶段先用均匀数据训练约 1500 万参数的参考模型并冻结，再以相同规模、重新初始化的代理模型逐域计算损失差：代理损失减参考损失。代理最小化由域权重加权的差值；权重则以指数梯度上升追逐最差差值，并加入熵奖励，避免概率集中到少数域。最终权重取末尾 1 万步的平均。第二阶段把权重固定，用各域加权行为克隆损失训练目标 VLA。代理和参考训练 10 万步，每域批量 128；代理学习率为 1e-4，权重更新率为 1e-2，熵系数 λ=0.1。紧凑模型用视觉历史和语言指令作为输入，经视觉/语言编码器与六层跨注意力解码器预测离散动作序列。

#### 方法对比分析
均匀混合不区分数据质量；按域大小采样还会放大大域影响。普通 group DRO 用绝对域损失，可能过度追逐噪声；Re-Mix 采用参考损失作基线，但搜索也依赖较大模型。ProMix 保留 excess-loss 思路，用小代理学习权重，从而把搜索与最终骨干规模分离。它和轨迹级筛选互补：只调整整域贡献，不剔除域内单条示范。

#### 实验分析（精简版）
实验使用 9500 条 Open-X 回合，七个机器人/任务域，每域 80/20 训练验证划分；测试在实体机器人上对 15 个留出任务各做 20 次闭环试验，平均五个随机种子。50M 模型上，ProMix 平均成功率 76.2%，均匀混合为 68.4%；最差域从 51.4% 提升到 68.5%。与 Re-Mix 的 74.8% 差距落在置信区间重叠范围内。去掉参考模型后平均值为 72.4%，去掉熵正则后最差域为 48.7%，显示两组件都影响配比质量。15M 权重迁移到 7B 仍有提升，但该实验采用 OpenVLA 架构，兼有规模与架构变化；数据范围也限于一个子集。

#### 实用指南
论文报告的搜索账本为 11 GPU 小时：参考训练 3 小时、代理搜索 8 小时；对照的全规模搜索在 50M 为 16 小时、7B 为 192 小时。这里不含评估、超参网格搜索和最终策略训练，也未对硬件做归一化。复现需保持代理与参考的域批次可比，冻结参考，验证 λ，并检查权重分布与最差域指标。迁移到新语料时，域划分和参考模型质量都可能改变权重；可与样本级筛选组合，但论文未验证这种组合。

#### 总结
小代理学习可迁移配比

1. 按机器人或任务整理示范域。
2. 训练并冻结均匀混合参考模型。
3. 代理按域比较损失差，鲁棒更新权重并用熵防塌缩。
4. 固定权重训练更大的目标策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.29873v1)
- [arXiv](https://arxiv.org/abs/2611.29873v1)

---

<a id='2611.30022v1'></a>
## [MOVER: Planning Beyond Motion](https://arxiv.org/abs/2611.30022v1)

**Authors:** Xiangyu Li, Yu Jiang, Tianyi Wang, Liang He, Qixing Huang, Zhaomiao Guo

**Published:** 2026-11-23

**Categories:** cs.CV, cs.RO

**Abstract:**

Residual world models forecast a driving scene through its temporal change, so feature components that stay constant over time vanish from the residual pathway. We ask whether evidence with weak temporal change, such as a waiting pedestrian or a stopped vehicle, can be used more effectively for planning, and introduce MOVER, a complementary event-routing mechanism. A training-time teacher measures how each spatially grounded event query responds when a past frame is removed from shared history and contrasts that sensitivity with the event's pooled temporal residual, prioritizing events that depend on history but receive weak residual evidence. A student predicts the resulting priorities from unperturbed history, so the teacher adds no inference cost. Under fixed total routing mass, one distribution scales a low-rank future bird's-eye-view update and biases trajectory-conditioned event reading. On nuScenes, MOVER reduces ResWorld's UniAD-style average L2 from 0.65 to 0.55 m and its collision rate from 0.23% to 0.12% without direct planner ego status (0.50 m and 0.09% with it), adding 4.1 ms per frame. Matched controls show that the complementary teacher outperforms either signal alone and that the allocation acts mainly during training. Adapted to BridgeDrive-LEAD, MOVER reaches a 96.52 +/- 0.65 driving score over three seeds on the 220-route Bench2Drive closed-loop benchmark, exceeding matched controls in each paired seed by a margin within seed variation.

### 论文解读

#### 摘要翻译

残差世界模型预测场景的时间变化，因而静态特征较少进入残差分支。MOVER训练教师删除一帧历史，比较空间事件表示变化与时间残差，为依赖历史但残差弱的事件分配较高优先级。学生从完整历史学习优先级，推理无需教师。路由同时调节未来鸟瞰图更新和轨迹解码的事件读取。nuScenes规划指标改善；闭环结果高于匹配对照，但统计不确定性较大。

#### 方法动机分析

等待行人、停驶车辆等上下文即便当前没有明显运动，也可能限制转弯或通行。时间差分容易弱化静态内容，但仅凭残差较小不能判断一个区域是否有用：它可能包含历史信息，也可能只是无关背景。MOVER据此假设，“删除历史后表示改变”与“局部净残差弱”结合，比任一信号单独使用更适合挑选补充证据。事件是空间摘要，不是对象框。

#### 方法设计详解

输入为自车坐标对齐的当前及历史 BEV。模型从当前帧形成八个带空间先验的软掩码，并复用于历史帧池化和交叉注意力，得到增强事件。训练时随机移除一个历史帧、保留当前帧，以事件表示的变化量计算历史敏感度；同时在掩码区域内先汇集最近两帧差、再计算平方范数，作为净残差。逐场景标准化后，以敏感度减去残差的加权值形成互补分数，并经 softmax 得到教师分布。学生路由器通过 KL 损失拟合教师，但输入未扰动历史。

推理时，学生产生的概率被换算成总量固定的权重。同一权重一方面缩放事件特征，将低秩补偿写入基线预测的未来 BEV；另一方面作为轨迹查询读取事件上下文的先验。轨迹解码器据此输出有界位移修正。训练还使用路径位置损失及依赖未来演员标注的 clearance 损失；这些标注不会进入推理。默认事件数为 8、β=1、路由总量系数 ρ=0.25。

#### 方法对比分析

相比只预测时间残差或按运动强度门控的方法，MOVER将删帧敏感度与净残差组合起来，补充未来场景和轨迹读取；它保留所有事件，只重分配固定路由总量。其核心创新在训练时用模型内部扰动形成事件优先级，无需对象重要性标注；但该分数只能说明模型表征依赖历史，不直接说明事件对规划的因果作用。

#### 实验分析（精简版）

在nuScenes开环评测中，MOVER的UniAD-style平均L2为0.5528 m，低于ResWorld的0.65 m；碰撞率为0.1172%，ResWorld为0.23%。匹配消融中，双路径学习路由L2为0.5528 m，均匀路由为0.6285 m，互补教师优于单信号目标。RTX 3090每帧新增约4.1 ms。Bench2Drive 220路线三种子得分为96.52±0.65，对照均匀路由为95.92±0.40，差异区间跨零。nuScenes没有多种子方差，碰撞事件数少，静态演员分组也未证明对象级强化，因此结果支持规划表现改善，不足以确认安全收益或静态对象因果解释。

#### 实用指南

复现需对齐BEV并在历史间共享掩码；训练随机删帧形成教师，再蒸馏至单次前向学生。作者使用AdamW基础学习率1e-5，路由器和轨迹修正器倍率为10、30。迁移到其他时序规划器时，应重新验证事件分辨率、残差窗口和扰动策略，并补充多种子及反事实对象实验。

#### 总结

历史敏感与弱残差互补路由

1. 对当前及历史BEV用共享空间掩码池化事件。
2. 训练时删一帧历史，比较表示敏感度与净残差。
3. 将互补教师分布蒸馏给完整历史上的学生路由器。
4. 用固定总路由量补写未来BEV并引导轨迹读取。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.30022v1)
- [arXiv](https://arxiv.org/abs/2611.30022v1)

---

<a id='2611.31209v1'></a>
## [WorldLatent: A Unified 4D Latent Space for World Perception and Generation](https://arxiv.org/abs/2611.31209v1)

**Authors:** Jingxi Chen, Yujia Chen, Cornelia Fermuller, Yiannis Aloimonos, Wen-Sheng Chu, Du Tran

**Published:** 2026-11-23

**Categories:** cs.CV

**Abstract:**

Generative Pre-Training (GPT) represents a profound milestone in natural language processing, serving as the foundation for most of today's state-of-the-art large language models. By casting diverse understanding and generation problems into a shared token representation, GPT enables a single architecture and unified optimization loop to master heterogeneous tasks across massive corpora, giving rise to powerful foundation models for natural language. In contrast, 4D spatiotemporal vision lacks a common representational protocol: perception (such as depth and camera estimation) and generation are either handled by separate one-way pipelines or bolted together from unaligned single-modality spaces. In this work, we introduce WorldLatent, a framework that formulates a unified 4D latent space as a common protocol for both world perception and generation. Rather than relying on fragmented, task-specific models or stacked single-modality representations, our framework proposes a multimodal VAE architecture that compresses and fuses complex 4D visual appearance, geometry, and dynamics into a single structured latent representation. Within this formulation, perception and generation operate as complementary denoising processes over the same latent space, enabling one generative model to learn both capabilities simultaneously within a single training loop. Experiments across representative 4D benchmarks verify that our fused latent protocol achieves specialist-level zero-shot 4D perception, coherent 4D generation, and controllable editing from a single model. By demonstrating that diverse 4D tasks can be seamlessly unified under a shared representational protocol, our work provides a scalable foundation for general-purpose 4D spatial intelligence.

### 论文解读

#### 摘要翻译
WorldLatent提出统一的4D潜空间，将视频外观、场景几何和相机运动编码为共同表示，让一个生成模型在同一训练循环中执行4D感知、场景生成和可控编辑。作者在多个基准上展示了零样本几何估计、文本生成4D视频和深度控制重渲染能力。

#### 方法动机分析
现有几何模型通常只输出深度、点云或相机姿态；视频扩散模型则重视逼真像素，缺少显式三维约束。把几何信号简单接入冻结RGB编码器，或拼接彼此独立的潜变量，会造成冗余且难以统一训练。作者假设，外观、深度和运动若共享紧凑的时空潜表示，感知与生成便可成为同一目标的条件去噪问题。

#### 方法设计详解
场景被表示为四个逐像素对齐的视频张量：RGB、深度、相机射线方向及射线矩（由相机中心与射线方向叉乘得到）。四个模态编码器分别提取特征，DPT在多个ViT层融合对应时空位置的特征，形成32通道的WorldLatent；四个解码器从该潜变量重建RGB、深度和射线。训练损失结合各模态L1重建、RGB感知损失、深度梯度、点图几何误差和KL正则，并随机屏蔽输入模态以支持不完整条件。之后冻结VAE，训练一个扩散Transformer预测完整4D潜变量。它把带噪目标与可见模态编码沿通道拼接，文本经交叉注意力注入。输入RGB时，输出解码为深度与相机轨迹；输入文本时，从噪声共同生成视频、深度和运动；输入深度加文本时，则在几何约束下编辑外观。训练任务按文本生成0.45、深度编辑0.30、视频感知0.25采样。最终14B模型在256个32GB TPU上训练108k步，批量256；推理分别用25步生成/编辑和8步感知。

#### 方法对比分析
相较于将RGB、深度、射线分别编码后拼接，WorldLatent用DPT融合四类信号并让所有任务预测同一完整潜变量。它不是为每个任务另设模型，而是通过可见模态子集切换任务。该设计适合需要同时构造场景几何和外观的4D建模；代价是共享模型仍需足够容量，且各模态的监督质量会共同影响潜空间。

#### 实验分析（精简版）
WorldVAE使用5,120个DiT token，与Wan RGB VAE相同；分离模态需要15,360个token，单步计算量由148.1升至569.1 TFLOPs。Sintel零样本深度评测中，8步WorldLatent的δ1准确率达到0.769，高于DA3-Giant的0.763；但其AbsRel为0.247，差于教师模型0.194，表现并非全面领先。Bonn、ScanNet也纳入感知测试。55个留出视频提示上，模型生成相机转角30.45°、动态分数0.964，相比Wan基线的0.99°和0.400更有运动；此生成评测规模有限，且训练真实视频的几何标签大量由DA3-Giant伪标注。1.3B消融显示统一模型接近单任务模型，但作者未验证工业规模扩展。

#### 实用指南
论文给出了训练数据比例、模型规模和推理步数，但页面仅链接项目网站，未在文中明确说明代码或权重的公开许可与下载方式。复现需约140万段17帧视频，覆盖真实视频、合成4D及标注扫描数据，并为无密集真值的视频生成深度和相机伪标签。迁移至机器人或自动驾驶时，可将传感器观测作为部分模态输入，利用共享潜变量做几何估计与条件场景编辑；仍须用标定传感器数据和下游任务验证度量精度、安全性与动态遮挡鲁棒性。

#### 总结
核心思想：让感知与生成共享4D空间。
1. 对齐视频、深度与相机射线。
2. 用多模态VAE融合为统一潜变量。
3. 随机屏蔽模态，训练扩散模型预测完整场景。
4. 用RGB、文本或深度加文本切换感知、生成和编辑。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.31209v1)
- [arXiv](https://arxiv.org/abs/2611.31209v1)

---

<a id='2611.29896v1'></a>
## [Segment Anything In-Context via Visual Demonstrations](https://arxiv.org/abs/2611.29896v1)

**Authors:** Yuzhu Wang, Manni Duan, Hongliang Ye, Shu Kong

**Published:** 2026-11-23

**Categories:** cs.CV

**Abstract:**

We present the Segment Anything Model via Visual Demonstrations (SAYD), a foundation model for in-context segmentation. Given visual demos comprising prompt images and their segmentation masks, SAYD segments coherent regions (e.g., objects or their parts) in a query image by establishing visual correspondences between the prompts and the query. SAYD generalizes to previously-unseen segmentation tasks, including those requiring tacit visual knowledge or involving concepts that are difficult to articulate unambiguously in language. We train SAYD on 12 public datasets originally curated for panoptic segmentation using our proposed techniques, including a lightweight mask decoder for prompt-agnostic mask generation and a learning mechanism for recognizing novel query classes absent in the prompts. These innovations enhance both computational efficiency and out-of-domain generalization. Consequently, SAYD significantly outperforms existing methods of in-context segmentation and open-vocabulary segmentation across 24 out-of-domain benchmarks, spanning natural, biological, ecological, cosmological, medical, pathological, and satellite imagery.

### 论文解读
#### 摘要翻译
本文提出视觉示例分割模型SAYD：给定带掩码的示例图，模型在查询图像中找相似区域。它面向细胞、星系或材料等难以用语言描述的目标。作者用12个公共数据集训练，在24个跨域基准上测试。

#### 方法动机分析
文本提示难表达专家凭经验辨认的细微形态；通用特征也未必能精确匹配像素。作者假设示例定义目标，而查询图决定边界。因此分开生成候选掩码和识别示例类别，并训练模型处理示例未覆盖的类别。

#### 方法设计详解
输入为带掩码的提示图和查询图。ViT-L提取特征，提示掩码区域平均池化成类别原型。查询特征进入轻量解码器，200个mask query经4层Transformer生成候选掩码；掩码仅由查询特征决定。query token与提示原型点积分类。训练采用带可学习温度、偏置的sigmoid BCE，将示例未覆盖类别作为已知类负例；推理时最大校准logit低于0即拒识。总损失为20倍BCE、2倍focal和dice。训练用约91万图像、1146万掩码；每样本5张提示图加1张查询图，512×512，训练55万迭代。

编码器使用DINOv3 ViT-L初始化，并在训练中微调；消融显示冻结编码器会明显降低示例分割效果。作者还发现，去掉温度会导致训练失败，加入可学习偏置也能改善分类校准。推理时先把同类候选掩码合并，可输出语义或全景结果。提示图的特征只需编码并汇总成原型，因此增加示例主要增加原型计算，而无需为每类重新生成查询掩码。

#### 方法对比分析
相较于提示特征参与掩码生成的方案，SAYD先生成类别无关掩码，再按原型分类，避免按类别重复解码。它适用于难以命名但能用视觉例子指定的目标；高度不规则边界可能难以处理。

#### 实验分析（精简版）
24个OOD基准覆盖语义、实例和全景分割。Cell语义mIoU为22.43，SAM3为0.00；Ant上为32.04对9.36。一个查询/提示对需0.3T FLOPs、峰值显存1.6GB；类别数1增至1000时耗时约0.101至0.107秒。SAM3在常见物体仍有竞争力，Fashionpedia实例AP为34.93，高于SAYD的27.60。极不规则形状和每类单原型是局限；每基准约10张提示图覆盖全类别，低提示量表现还需验证。

#### 实用指南
论文提供自包含代码和评测、预测、特征可视化notebook，但未随稿附模型权重，称将以MIT许可发布代码与模型。复现时需关注训练数据许可与版本、提示集构造及三组提示平均协议；训练使用DINOv3 ViT-L初始化、batch 128、AdamW和55万迭代。迁移至机器人视觉时，可将少量带掩码的相机帧作为示例，探索跨场景长尾物体或部件分割；这属于迁移推断，论文未验证时序稳定性或实机部署。

#### 总结
核心思想：示例定义类别，查询决定边界。
1. 提供带掩码的视觉示例和查询图。
2. 从示例掩码汇总类别原型。
3. 查询特征独立生成候选掩码。
4. 用原型分类，并拒识未见类别。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.29896v1)
- [arXiv](https://arxiv.org/abs/2611.29896v1)

---

<a id='2611.30193v1'></a>
## [Visual-RFT++: Extending Visual Reinforcement Fine-Tuning to Multimodal Tool Use](https://arxiv.org/abs/2611.30193v1)

**Authors:** Ziyu Liu, Zeyi Sun, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, Jiaqi Wang, Yuhang Zang

**Published:** 2026-11-23

**Categories:** cs.CV

**Abstract:**

Reinforcement Fine-Tuning (RFT) has recently become a central paradigm for advancing Large Reasoning Models (LRMs), as demonstrated by systems such as OpenAI's o3 and the R1 series. Despite significant progress in LLMs, how to leverage RFT to unify visual perception, step-by-step reasoning, and agentic tool use into one coherent training paradigm remains an open challenge. In this work, we present Visual Reinforcement Fine-Tuning++ (Visual-RFT++), a unified framework that extends RFT to both visual perception tasks and multimodal agentic workflows, enabling LVLMs to reason with images, act upon images, and evaluate their own outputs using verifiable reward. With Visual-RFT++, open-source LVLMs not only show significant gains on diverse visual perception tasks, but also acquire agentic abilities such as web searching for real-time information and code-based image manipulation (e.g., cropping, rotation, and analysis). Across visual perception tasks, including fine-grained classification, few-shot detection, reasoning grounding, and open-vocabulary detection, Visual-RFT++ delivers strong gains over SFT, including a +24.3 percentage points improvement in one-shot fine-grained classification and +21.0 percentage points on COCO's 4-shot setting and +15.4 percentage points on LVIS with around several hundred samples. To evaluate agentic behaviors, we introduce the Multimodal Agentic Tool Bench (MAT) with MAT-Search and MAT-Coding. Visual-RFT++ surpasses its baseline by +18.6 F1 / +13.0 EM percentage points on MAT-Coding and +10.3 F1 / +8.7 EM percentage points on MAT-Search. It also shows gains of 29.3 F1 and 25.9 EM percentage points on multi-hop QA benchmarks such as 2Wiki and HotpotQA, demonstrating strong cross-modal generalization. Visual-RFT++ introduces a data-efficient and reward-driven framework that strengthens both visual reasoning and agentic adaptability.

### 论文解读
#### 摘要翻译
Visual-RFT++把可验证奖励驱动的强化微调用于视觉感知与多模态工具代理。模型在少量样例上学习分类、检测、定位，也学习搜索网页或编写代码处理图像。作者报告其在多项视觉任务及新建MAT工具基准上优于监督微调，并在文本多跳问答上展现迁移表现。

#### 方法动机分析
监督微调通常需要逐例整理标准答案，少样本时容易过拟合或退化。作者希望让模型自行提出多种解法，再依据可自动核验的结果调整策略。核心假设是：分类正确性、框与目标重叠、检索查询相关性和最终答案质量都能构成有效奖励，从而用较少标注学会推理与工具使用。

#### 方法设计详解
输入图像与问题后，LVLM生成多条带思考和答案的响应；GRPO按同组奖励的相对高低更新策略，并用参考模型的KL约束稳定训练。检测奖励综合框与真值的IoU、置信度和格式；分类用类别匹配与格式奖励。代理任务采用结构化的think、search、code、answer标记：搜索模型分解多跳问题、查询网页、读取返回证据并继续推理；编码模型判断图像是否需要旋转、裁剪或增强，再生成Python代码交给环境处理。答案以F1评分，搜索查询以语义相似度评分，代码块获得动作格式奖励。多轮交互拆成带此前记录的逐轮样本，每条只奖励当前动作。训练时使用Qwen视觉语言模型，AdamW学习率1×10⁻⁶，8张GPU；每个提示采样8个响应，最大生成2048 token，BF16和ZeRO-3训练，KL系数0.04。

#### 方法对比分析
与监督微调模仿固定答案不同，Visual-RFT++优化模型采样出的行为；与只接入搜索或代码工具的提示方法相比，它还训练何时调用工具、怎样组织查询以及如何利用反馈。其新意主要是把视觉感知奖励和多模态代理奖励放入同一GRPO范式，并提出MAT-Search、MAT-Coding评测。奖励并非真正通用：不同任务仍需设计可核验的规则，代码内容也没有被参考程序逐项监督。

#### 实验分析（精简版）
四个细粒度分类集的1-shot平均准确率为80.3%，较未微调基线56.0%高24.3个百分点；SFT为51.7%。Qwen2.5-VL-7B在MAT-Coding的平均F1/EM由32.12/21.50升至50.68/34.50；MAT-Search由53.49/46.67升至63.77/55.33。相同骨干和工具条件下也优于ReAct或代码执行提示。局限是MAT评测规模较小，图像编码集基于合成畸变，且多数感知实验没有多随机种子结果；搜索依赖外部检索服务。它们证明基准上的改进，尚不能证明真实部署稳健性。

#### 实用指南
论文称代码可用。复现代理训练需准备多轮工具环境、缓存搜索结果和可核验标注；报告的设置是8卡、每提示8条采样、学习率1e-6。感知端需针对任务实现奖励，例如检测框匹配与格式校验。可迁移到少样本视觉识别、图像质量修复代理或需要检索的多模态问答；机器人场景还需加入时序动作、安全约束和实机闭环验证。

#### 总结
核心思想：让模型从可核验反馈中学会看与行动。

1. 输入少量图像问题，采样多种带推理的回答。
2. 用任务规则检查答案、定位框或工具动作。
3. 以GRPO奖励较好的轨迹，并限制策略偏移。
4. 在搜索或图像处理环境中接收反馈，再完成回答。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.30193v1)
- [arXiv](https://arxiv.org/abs/2611.30193v1)

---

<a id='2611.30195v1'></a>
## [Quantization for Action, Execution for Deployment: Joint Precision and Execution Selection for World Action Models](https://arxiv.org/abs/2611.30195v1)

**Authors:** Tianqi Li, Xin He, Xue Geng, Le Zhang, Ming-Ming Cheng, Yun Liu

**Published:** 2026-11-23

**Categories:** cs.CV

**Abstract:**

World action models guide robots by predicting the effects of actions or generating commands from observations. A model may compare hundreds of candidate futures before acting, repeatedly reading weights that must fit in device memory. Post-training quantization reduces the number of bits used to store pretrained weights, but a smaller model is not necessarily a deployable one: temporary tensors can exceed graphics processing unit (GPU) memory limits, while even small prediction errors can change the selected action. This creates two coupled challenges: identifying bit assignments that preserve task performance and determining which of them meet real execution constraints, without exhausting costly task trials. We present PTQ4WAM, a framework for choosing layer-wise bit assignments and execution routes under memory, latency and task-evaluation budgets. Calibration measurements guide candidate proposals, optionally with a language model agent. We profile full-model routes on the GPU before spending task trials on feasible assignments. Fusing group rescaling and reading visual history in place can make an otherwise infeasible assignment deployable. Across 15 workloads from four model families, the median task score is 99.2% of the corresponding 32-bit floating-point (FP32) score. On fixed GPU replays, the median speedup over FP32 is 2.36 times and the median reduction in peak allocated memory is 71.5%. The code will be released soon.

### 论文解读

#### 摘要翻译
世界动作模型通过预测动作后果或直接生成控制命令来指导机器人。它们可能要反复读取权重并比较大量未来，因此既受显存和延迟限制，也容易因量化误差改变动作选择。PTQ4WAM联合决定逐层权重位宽与GPU执行路线：先用校准统计提出候选，在目标设备上检查完整模型调用是否可运行，再把有限任务评测用于可行候选。15个工作负载、4类模型的实验中，任务分数中位数达到对应FP32的99.18%，固定输入GPU重放中位速度提升2.36倍，峰值分配内存中位下降71.46%。

#### 方法动机分析
常见量化指标关注权重或特征重建误差，但规划器最终比较候选动作，微小误差若翻转排名就可能改变控制结果。另一方面，模型文件变小并不代表运行时显存更低：组缩放的中间张量和历史特征拼接副本都可能造成峰值超限。作者据此把精度图与算子路线作为联合决策，并假设校准数据足以筛选候选、完整GPU剖析能在任务试验前排除不可部署方案。

#### 方法设计详解
输入是预训练WAM、原任务规划器/控制器、校准样本、存储预算和设备内存/延迟上限。系统为可量化模块构造4/8位精度图，将尺度、受保护层和文件头等纳入准确文件大小；可选语言模型根据模块结构和校准统计提议不同映射，确定性生成器再检查约束。随后对每种映射测试GPU路线，记录全调用的峰值进程内存与尾延迟，并验证数值结果；仅可行映射进入原任务的限额评估，最终按源等权任务得分选出模型和路线。

执行优化包括两项。FGR把每个输入组的整数部分和乘以对应尺度，融合进生产该结果的线性算子，避免物化完整临时输出；不同组尺度必须分别应用，且累加器不能溢出。DHR令消费者直接按逻辑位置读取驻留历史片段，省去连续拼接；它要求片段保持不变，并保留位置、掩码和对整个历史统一归一化。减少局部张量只有在它参与决定全模型峰值时才会降低峰值。

#### 方法对比分析
重建误差或曲率分数评估局部近似；PTQ4WAM则让实际规划器/控制器评估完整精度图的任务结果，同时把硬件执行路线纳入可行性。与先评分后检查资源的流程相比，它避免把任务试验花在运行超限的模型上。FGR/DHR在精度固定时单独改执行实现，因此可以区分位宽收益与临时内存优化。该思路适用于反复推理、规划反馈昂贵且部署资源上限明确的模型。

#### 实验分析（精简版）
15项中位任务分数恢复为FP32的99.18%，中位速度提升2.36倍；内存中位下降71.46%。在LeWM Wall，PTQ4WAM与匹配噪声的特征误差相同（0.0225），但动作一致率为93.4%对81.9%、成功率为94.25%对79.87%，支持直接按任务决策评价映射。局限是硬件测量集中于RTX 4090，9项任务分数仍下降；记录轨迹评价也不等于实机验证。准备和剖析成本可能很高。

#### 实用指南
论文称代码即将发布，暂未给出可复现仓库。复现需固定模型检查点、任务源划分、候选池和序列化预算；在目标GPU上测量完整调用的进程峰值和尾延迟，不能只测单个算子或权重文件。Long案例使用RTX 4090，48个映射、每个两条路由各20次预热和100次计时；完整准备约7309.5秒。换硬件、历史长度或输入形状后需要重做路由测量。迁移到机器人/自动驾驶可把规划候选规模、历史缓存和时限放入工作负载定义，但属于基于框架的推断，尚无跨硬件或实车验证。

#### 总结
核心思想：任务选精度，设备选路线

1. 用校准数据和预算生成合法低比特候选。
2. 在目标GPU测全调用，筛除超内存或超时路线。
3. 用原任务反馈挑选可行精度图，按需分阶段扩展评测。
4. 以FGR融合缩放、DHR直读历史，保存并部署选定组合。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.30195v1)
- [arXiv](https://arxiv.org/abs/2611.30195v1)

---

