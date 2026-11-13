time: 20261113

# Arxiv Computer Vision Papers - 2026-11-13

## Executive Summary

## 执行摘要

本期十篇论文集中在具身智能与机器人学习，同时覆盖驾驶视频生成、三维场景图和视觉姿态估计。一个共同主题是：只看总体预测分数或相机轨迹，可能掩盖模型是否真正响应动作计划、能否跨视角维持实体关系，以及仿真结论能否对应真实执行。多篇工作通过配对对照、真实数据参考或细分诊断来补足评估。

较值得优先阅读的是《Driving World Models Move the Camera, Not the Scene》：它发现驾驶世界模型会跟随相机而忽视场景变化，并以日志重定时、立体观测和 ReTime-Compose 检查计划响应，证据仍属离线评估。ActionCodec2 通过逐步残差编码和流式解码连接动作压缩与实时执行，在模拟基准及四项真机任务报告结果。跨模拟器世界模型论文展示了模型路由用于多种模拟变化的潜力，但真实 ALFA 数据只验证飞行预测。另可关注持久性对照诊断、RSLiveBench-UAV 与 Genie Sim：它们从不同角度说明，评估设计本身会影响我们对模型能力的判断。

整体方向正从单一基准分数转向更有针对性的能力检查：动作扰动测试动力学模型对未见动作的敏感性；成对真实—仿真任务衡量模拟器对应程度；增量场景图将对象召回与定位误差分别报告；零样本姿态估计则把推理速度与精度同时纳入比较。读者可先读驾驶世界模型与 ActionCodec2 获取方法细节，再按研究兴趣选择基准和诊断类论文，并留意模拟结果、离线回放和闭环实机证据之间的差别。

---

## Table of Contents

1. [Driving World Models Move the Camera, Not the Scene](#2611.15260v1)
2. [ActionCodec2: A Streaming Action Tokenizer](#2611.15490v1)
3. [From a Single RGB Image to Robot Execution: Metric Scene Alignment and Contact-Guided Retargeting](#2611.14981v1)
4. [Prediction or Persistence? What Future-Prediction Scores Miss About World Models](#2611.15082v1)
5. [AFTER: A World Model with Adaptive Fast--Slow Timescales for Error-Driven Retention](#2611.15215v1)
6. [Does Your World Model Know Actions Have Consequences?](#2611.15048v1)
7. [RSLiveBench-UAV: A Continually Evolving Vision-Language Benchmark for Aerial Spatial Intelligence](#2611.14953v1)
8. [Genie Sim Benchmark: Real-to-Sim Fidelity and Evaluation](#2611.15020v1)
9. [IncR-3DSGG: Incremental Posed RGB Image Based 3D Scene Graph Generation via Privileged Depth Supervision](#2611.15429v1)
10. [Fear of Memorizing Objects: End-to-End Zero-Shot Model-Based Object Detection](#2611.15459v1)

---

## Papers

<a id='2611.15260v1'></a>
## [Driving World Models Move the Camera, Not the Scene](https://arxiv.org/abs/2611.15260v1)

**Authors:** Yuqiao Feng, Xiaoyu Ao

**Published:** 2026-11-11

**Categories:** cs.CV

**Abstract:**

Plan-conditioned driving world models are used to compare candidate plans, so their forecast should change with the plan as the real scene would, yet widely used control metrics check only whether the camera follows the command. Recorded drives already contain real outcomes of plans that were never driven: retiming a log gives pose-matched images and LiDAR for other speeds, and a stereo rig records a laterally shifted view. Against these references, released world models move their camera but barely move the scene: their scene response, the share of the real change in distance that the forecast reproduces, reaches at most about a third of real video's, and the trajectory score cannot tell an executed speed command from an ignored one. Interventions inside the generator trace the failure to the plan's channel. The forecast geometry follows the context frames; once they carry the planned geometry, plan tokens add little. ReTime-Compose therefore delivers the plan as geometry, without training. It transports the frames of one logged-speed rollout to the planned cameras, and the frozen model completes only what no frame covers. Frozen generators then follow a 0.54 m lateral shift nearly as the real scene does on KITTI, nuScenes and Argoverse 2, and respond more than re-sampling to slow-down plans in three generators and to speed-up plans in two.

### 论文解读
#### 摘要翻译
驾驶世界模型的预测画面可能跟随相机指令，却没有让道路场景按计划改变。作者用重定时行车记录和立体相机作反事实参考，提出诊断指标与无需训练的 ReTime-Compose。 这项工作提供的是可复核的研究结果与方法边界；理解其贡献时应同时看任务设置、观测来源及评价对象。

#### 方法动机分析
驾驶规划器会比较多条候选轨迹，若生成视频只响应相机姿态、不响应车速或横向计划，常用轨迹指标可能误判模型可信。论文的动机是检查计划是否真的改变了场景内容。 关键假设在于现有评估或建模流程遗漏了任务中的重要变量，因此作者把该变量显式纳入流程，并用针对性实验验证。

#### 方法设计详解
训练集用于拟合或构建论文所需组件，推理时按同一任务协议生成预测；训练和推理设置应分开理解。先用日志中的一次实际车速滚动生成视频，再根据目标速度和相机姿态重定时图像与深度。将真实帧几何变换到计划相机位置；冻结生成模型只补全没有被任何帧覆盖的区域，保留可直接对应的观测内容。作者另以距离变化定义场景响应比例，并通过干预生成器区分上下文几何与计划token的作用。 从输入到输出的完整流程需要保持各模块接口一致；论文的设计将数据处理、核心预测与评价连接起来，使贡献能够通过消融或对照定位。

#### 方法对比分析
相较简单重采样或只看相机轨迹误差，该方法把计划几何显式送入上下文，并尽可能复用真实观测。它不训练新生成器，便于诊断现有模型；优势依赖可靠深度、位姿和足够帧覆盖，且合成视频仍可能有补全伪影。 因而读者应把它看作特定设置下的改进，比较时需要保持数据、指标和预算一致，避免从单一总体指标外推到未测试场景。

#### 实验分析（精简版）
在 KITTI、nuScenes、Argoverse 2 的日志与立体参考上，0.54米横向偏移时场景响应约为真实变化的0.93至1.04倍；对减速计划，在三个生成器上优于重采样0.11至0.24，增速计划在两个生成器上改善。结果是离线参考一致性，不是闭环驾驶成功率。 这些数值描述论文所测范围，评价协议和样本构成会影响绝对分数；作者结果支持方法在本设置下有效，尚需独立数据复验。

#### 实用指南
复现需对齐相机内外参、深度、时间戳及日志重定时方式，并报告几何覆盖率。论文发现位姿比例会影响结果；迁移到新车载相机前应分路线和速度范围验证，同时人工检查补全区域与动态目标边界。 实施时建议先复核数据许可与预处理，再用小规模样例确认输入输出、随机性和指标实现；保留逐项结果有利于发现迁移退化。

#### 总结
核心思想：把计划几何写入生成上下文。
1. 用真实日志构造计划条件参考。
2. 变换已观测帧，只生成未覆盖区域。
3. 比较场景响应并检查适用条件。 结论是有边界的经验贡献，应结合具体需求决定是否复现。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.15260v1)
- [arXiv](https://arxiv.org/abs/2611.15260v1)

---

<a id='2611.15490v1'></a>
## [ActionCodec2: A Streaming Action Tokenizer](https://arxiv.org/abs/2611.15490v1)

**Authors:** Zibin Dong, Yicheng Liu, Shiduo Zhang, Tianyuan Yuan, Baijun Ye, Jinqiu Wang, Hang Zhao

**Published:** 2026-11-11

**Categories:** cs.RO

**Abstract:**

Most action tokenizers for autoregressive vision language action (VLA) models compress motion into chunks that must be decoded in full before their actions become executable. We introduce ActionCodec2, a non-neural streaming tokenizer that releases actions incrementally while keeping trajectory reconstruction error within a prescribed tolerance. Its encoder quantizes each step relative to the decoder's reconstructed state, so the selected motion primitive corrects residual error rather than allowing errors from independently quantized increments to accumulate. Finding the smallest such primitive set is a geometric covering problem; we derive the exact minimum for translation and constant-factor-tight bounds for rotation. Set-BPE then losslessly merges co-occurring primitive symbols across coordinates and neighboring steps, without a fixed coordinate ordering. A universal version fitted on 20k hours of data from 24 embodiments averages 8.8 tokens per second for 15 Hz end-effector motion on this corpus. With the same VLA backbone, ActionCodec2 outperforms the compared tokenizers on three simulation benchmarks and in zero-shot real-robot evaluation. Its streaming representation allows policies to incorporate state feedback within an action chunk, substantially improving performance over open-loop execution under randomized dynamics, external disturbances, and moving objects. The same tokens also can serve as a direct control interface for VLM agents, which compose successful action sequences into reusable macros. These agent-built macros reduce control decisions and thinking-token usage across attempts and transfer to a held-out task. More results are available at https://zibindong.github.io/actioncodec2-page/

### 论文解读
#### 摘要翻译
本文提出ActionCodec2，一种面向机器人动作的流式分词器。它将动作拆成有界误差的运动原语，再用集合式字节对编码合并重复的时空结构，支持逐步重建和执行。作者在模拟基准与真实机器人上评估精度、速度和跨任务迁移。

#### 方法动机分析
动作分词需要同时满足压缩、可重建和低延迟。常见整体动作块或固定码本在长序列中容易累积误差，也会让机器人必须等完整动作块生成后才开始执行。作者的核心假设是：如果每个小步相对已重建状态编码，并显式控制残差误差，就能把误差限制在局部，同时保留在线生成能力。

#### 方法设计详解
输入为机器人连续动作轨迹。首先根据当前重建状态，把动作差分投影到一组固定运动原语，量化剩余残差；因此平移误差逐步受限，旋转则有常数因子界。其次，Set-BPE在空间与时间符号上学习可逆合并，允许不同次序的空间符号共享词元。解码端按时间收到词元后，即刻恢复原语和残差，并更新状态基准，再处理下一步。词表在多机器人数据上训练，覆盖24种具身配置。 训练时在24种机器人配置的数据上学习词表；推理端按收到的词元逐步解码，并在每步更新重建状态。论文报告的5毫米平移界是编码误差性质，不应理解为机器人实际轨迹误差界。

#### 方法对比分析
与FAST、OAT和X-Tokenizer等固定分块或全局码本相比，ActionCodec2的关键差异是逐步重建后再量化，因而控制单步误差，并把流式执行作为设计目标。它更适合需要边生成边控制、对轨迹复现敏感的任务；高动态动作可能不如整段编码紧凑，系统收益也依赖控制器能够利用高频反馈。

#### 实验分析（精简版）
在LIBERO上成功率98.2%，LIBERO-Plus为79.5%，SimplerEnv为93.8%。实时20Hz LIBERO执行为91%，分块基线为73.1%；真实机器人零样本任务进度为73.9%。实验支持低延迟与误差约束，但真机任务数仅四项，不能据此推断对不同机械臂、动作分布或长期运行均有同等收益。 这些任务中的模拟成功率、串流时延和真机进度衡量不同性质，数值不可直接横向相加；真实结果来自四项DROID-Franka任务。

#### 实用指南
适用于机器人动作序列的压缩传输、流式模仿学习和动作日志回放。复现时需统一运动原语、坐标系、控制频率及残差容差；作者报告使用约20k小时、24种机器人数据，15Hz采集和5mm/0.02rad容差。迁移到新平台时应先校准状态重建误差与执行延迟，并评估动态动作的压缩效率。 训练与推理的机器人坐标定义必须一致，执行时还要测量通信和控制器延迟。

#### 总结
核心思想：逐步编码并限制动作误差。
1. 将连续动作分解成固定运动原语。
2. 对相对重建状态的残差量化，并用Set-BPE合并符号。
3. 收到词元就重建、更新状态并执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.15490v1)
- [arXiv](https://arxiv.org/abs/2611.15490v1)

---

<a id='2611.14981v1'></a>
## [From a Single RGB Image to Robot Execution: Metric Scene Alignment and Contact-Guided Retargeting](https://arxiv.org/abs/2611.14981v1)

**Authors:** Zhenxuan Ding, Yu Huang, Xinmiao Wang, Chenghao Sun, Ziyang Dong, Wenhao Cui

**Published:** 2026-11-11

**Categories:** cs.RO

**Abstract:**

Extending humanoid interaction generation to new environments remains limited by the need for prebuilt 3D scene assets. Reconstructing scenes directly from single RGB images offers a way to overcome this limitation, but introduces a key challenge in maintaining geometric consistency throughout motion generation and transfer. Visually plausible human motions may be misaligned with the reconstructed scene, while retargeting can introduce additional support errors due to differences between human and robot embodiments.   We present a unified pipeline that transforms a single RGB image into a humanoid motion reference. It reconstructs scene geometry from the input image, generates a human interaction conditioned on the reconstructed scene, and retargets the motion to a humanoid robot. To maintain geometric consistency throughout this process, we incorporate two complementary refinement stages before and after embodiment transfer. Before retargeting, Joint Metric Interaction Refinement (JMIR) refines human motion alignment with the registered metric scene while preserving the generated interaction structure. Following retargeting, Contact Guided Residual Retargeting (CGRR) corrects robot specific support errors to restore the intended support relationships. Together, these stages produce motion references that are consistent with the reconstructed scene and can be followed by multiple pretrained humanoid trackers without scene specific policy fine tuning. Compared with GRAIL, our method reduces refinement time from 480 s to 87 s, yielding a speedup of 5.5 times. The complete pipeline achieves a success rate of 89% in simulation and 70% across 50 physical trials on the Unitree G1, demonstrating its effectiveness in generating executable humanoid motion references from single RGB images.

### 论文解读
#### 摘要翻译
本文研究如何将单张RGB图像中的人类动作转成机器人可执行运动。作者提出先对齐人和场景结构，再以接触与支撑约束修正机器人动作的两阶段流程，在模拟环境和Unitree G1实机上测试。 该系统关注感知与动作协同，并通过仿真和实体机器人实验检验其可行性。

#### 方法动机分析
直接把人体关节轨迹重定向到机器人，往往忽略两者的身体比例与场景支撑差异，导致脚滑、碰撞或失去平衡。单目重建还会带来场景几何误差。作者假设应先解决人和机器人的结构对应，再针对机器人实际支撑状态做局部修正，避免单一优化目标把两类问题混在一起。

#### 方法设计详解
输入是一张场景图像和生成的人类动作。SAM生成场景几何，人体动作先经JMIR对齐人体与机器人结构及场景关系，通过迭代优化保持主要交互和落脚位置。再由CGRR检查机器人接触与支撑，求解受约束的关节残差，使机器人恢复可行支撑。JMIR约迭代600次，CGRR每帧至多12次SQP；残差优化使用QP/SQP求解器。最终动作交给不同机器人策略执行。 方法把视觉观测、地形状态估计与动作生成连接为一个循环；控制器依据支撑与运动状态调整步态，训练策略则从仿真迁移到真实 Unitree G1。训练和推理环境差异是评估重点。

#### 方法对比分析
传统全身重定向通常一次性从人体姿态求机器人姿态，或依赖策略微调。本文分开处理跨形态结构对齐和接触可行性，后阶段可与多种执行器组合。更适合支撑主导的站立、行走或地面动作；对快速运动、移动物体互动以及细致手部操作的适用性尚未证明。 论文贡献主要在系统组合和实机验证，成功率不能与纯视觉识别或单一动作算法的指标直接比较。

#### 实验分析（精简版）
仿真成功率89%；Unitree G1在五种地形共50次实机试验中成功35次，即70%。无需微调时，三种执行策略成功率为50%至60%；对SONIC微调后达到90%，但使用约30小时数据。该结果表明流程可落地，但实机样本量有限，且依赖动作生成和场景重建质量。

#### 实用指南
复现需准备单目场景重建、人体动作生成、机器人模型、接触定义和求解器。论文设置JMIR约600次迭代，CGRR每帧最多12次SQP，并用QP求残差。迁移时重点验证地面与物体接触、机器人关节限位和场景尺度；不同执行器的表现不能只从姿态误差推断。 复现时需要记录机器人版本、地面摩擦与安全保护设置；仿真训练参数和实体推理频率都应分别公开。

#### 总结
核心思想：先结构对齐，再修正支撑。
1. 从RGB重建场景并生成动作。
2. JMIR对齐人体、机器人和场景结构。
3. CGRR根据支撑接触修正机器人动作并执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.14981v1)
- [arXiv](https://arxiv.org/abs/2611.14981v1)

---

<a id='2611.15082v1'></a>
## [Prediction or Persistence? What Future-Prediction Scores Miss About World Models](https://arxiv.org/abs/2611.15082v1)

**Authors:** Animesh Maheshwari

**Published:** 2026-11-11

**Categories:** cs.CV, cs.LG, cs.RO

**Abstract:**

Pretrained video world models are compared and chosen as components for robot learning largely by future-prediction scores. On its own, such a score cannot be read as evidence of prediction, because it mixes encoder quality, the near future's resemblance to the recent past, and the predictor's own contribution. We separate the three with a matched control. On 27,000 future-retrieval trials from household egocentric video, a model must pick the true continuation of a clip among eleven candidates from the same video (chance 0.091), ranked zero-shot by its own training criterion. Each model is paired with a persistence control, an identical copy whose predictor is never invoked, so the paired difference isolates the predictor. Score and contribution dissociate within the native-video V-JEPA family alone: the predictor adds +0.291 top-1 for V-JEPA 2.1 but +0.03 to +0.04 for V-JEPA 2 on the identical trials. Across five models the contribution ranges from +0.291 to negative, and it is negative only for the two action-conditioned models, which we evaluate outside their training regime. Persistence alone recovers the encoder ordering expected from pretraining, a sign that the probe tracks real signal. We suggest reporting each future-prediction score together with its persistence baseline. The gap between them, not the score, is what makes the number interpretable: a necessary check costing one forward pass. Establishing downstream usefulness is a separate matter, for a planning or control evaluation.

### 论文解读
#### 摘要翻译
本文提出一种世界模型预测评估诊断：对每个模型保留原预测器，再用相同模型关闭预测器作为持续性基线。两者分数差可估计预测模块超越“未来像现在”的贡献。作者在被动家庭视频的未来检索任务上比较多种模型。 研究提出一套诊断流程，以判断视频世界模型是否能在计划改变时预测持续结果。

#### 方法动机分析
未来预测分数常被当作模型能力指标，但静态场景或缓慢变化的视频，即使不预测也能靠复制当前状态获得高分。不同编码器和预测器的单独贡献因此难以区分。论文的核心假设是：保留模型其他部分、只关闭预测器，就能用配对差值隔离预测模块的增益或损害。

#### 方法设计详解
作者把评估拆成两个匹配流程：首先比较预测模型在同一锚点上的候选未来排序，再与结构相同但去除预测器的持续性对照配对。随后改变上下文长度、预测跨度和片段位置，检查预测器是否提供超越表征相似性的增益。测试覆盖约27,000次试验和1,094段视频；推理是离线候选选择，不包含在线控制。对照和预测器使用同一表征空间，便于把增益归因于预测模块。报告中还比较不同编码器和预测器组合，观察持续性基线是否已能解释模型表现。
#### 方法对比分析
相比只报告最终准确率，配对持续性控制成本低、无需额外训练，且能跨不同模型架构应用。它回答预测器是否带来增益，而不是模型是否有用或是否能控制机器人。对于动作条件预测器，还必须核实测试数据是否与动作分布匹配，否则负贡献幅度不宜直接比较。 与单看预测平均误差相比，干预测试直接检查因果响应；它是一种诊断，不提供可立即执行的控制器。

#### 实验分析（精简版）
27,000条试验来自1,094段、478.6小时家庭视频。V-JEPA2.1 ViT-g top-1为.736，持续性对照为.444，增量+.291；V-JEPA2 ViT-H仅增加.041。DINO-WM在被动视频上分布外，完整模型.136低于持续性对照.315。结果显示高预测分数未必代表预测器在工作。 在V-JEPA2.1上，持续性控制准确率0.736，对比0.444，差值0.291；V-JEPA2的提升仅0.041。DINO-WM在分布外测试为0.136，对照为0.315，显示不同模型的失败方式不同。

#### 实用指南
该诊断适合视频预测、世界模型和表征模型评测。需固定数据切分、候选窗口构造、原生距离函数和试验顺序，并在报告主分数时同步给出同模型对照。本文试验使用1–4秒上下文和未来窗口；迁移到机器人应改用与动作条件相符的可执行观测，不能把检索分数当控制成功率。 推理时应固定初始视频、动作扰动大小和预测时域，并按模型版本报告结果；数万次试验的统计单位及视频相关性值得单独核查。

#### 总结
核心思想：用持续性对照分离预测贡献。
1. 构造未来片段检索任务。
2. 原预测器和关闭预测器在同一试验上打分。
3. 报告配对增量并检查时间跨度。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.15082v1)
- [arXiv](https://arxiv.org/abs/2611.15082v1)

---

<a id='2611.15215v1'></a>
## [AFTER: A World Model with Adaptive Fast--Slow Timescales for Error-Driven Retention](https://arxiv.org/abs/2611.15215v1)

**Authors:** Haoyi Zhang, Qi Zhang, Yunfan Ren, Yi Bin, Lei Zhu, Heng Tao Shen

**Published:** 2026-11-11

**Categories:** cs.AI, cs.RO

**Abstract:**

In robotics and embodied AI, a core challenge for world models is adapting to sudden, persistent, yet initially unknown changes in robot dynamics during interaction. For instance, when a robotic arm grasps an object with an unknown payload, the robot dynamics change suddenly. If the world model fails to adapt, its prediction errors accumulate over the open-loop rollout and ultimately mislead the robot's actions. Existing world models, however, struggle to simultaneously satisfy three critical requirements for dynamics adaptation: responsiveness, reacting promptly to sudden changes to curb prediction error accumulation; stability, distinguishing persistent changes from transient disturbances to prevent incorrect updates; and persistence, retaining the adapted dynamics without new observations so that later predictions do not revert to the old dynamics. Inspired by predictive coding in the human brain with multi-timescale adaptation, we propose AFTER-WM, a world model with Adaptive Fast-slow Timescales for Error-driven Retention, whose prediction errors drive a fast correction that fades and a slow correction that persists, while dynamically adjusting the combination weights of pre-trained experts without altering them, to overcome the trade-offs among responsiveness, stability and persistence. On five simulated robot datasets and real UAV data, AFTER-WM has a lower multi-step mean error than all ten compared models and the highest mean return in a joint-failure control task. In simulation, its multistep mean error is 16%-34% below the strongest compared model's and is also the lowest 1-4 steps after a change and under transient disturbances alone, and its step-40 error is the lowest as well, reflecting responsiveness, stability and persistence. All code (method, baselines, experiments) and datasets will be released online.

### 论文解读
#### 摘要翻译
本文提出AFTER-WM，一种由预测误差驱动、能够快速适应并长期保留动力学变化的世界模型。模型通过快速状态、慢速状态和专家路由记忆分工处理突发扰动与持续变化，在多个机器人模拟环境和真实UAV故障飞行数据上评估。 论文提出为多种物理模拟器学习通用世界模型，并将其用于零样本控制。

#### 方法动机分析
单一适应速度存在冲突：反应快的状态容易受噪声影响，稳定的慢适应又会在动力学突变后反应太迟；长时预测中，短期修正还可能逐渐消失。作者假设将预测误差分配给不同时间尺度的记忆，并让慢状态和路由权重在预测期间保持稳定，可以同时改善突变初期和长时预测。

#### 方法设计详解
输入为当前状态、动作和最近上下文，预训练专家先预测下一状态。预测误差更新三种记忆：快速状态即时吸收变化并在开环预测中衰减；慢速状态以带不确定度的滤波方式积累持续偏差；路由记忆重新分配多个专家的权重。最终将专家预测与快慢校正相加。三类记忆只在上下文阶段接收新误差，进入预测后慢状态和专家权重固定。 训练流程由快速模型、慢速模型和路由器组成：路由器决定何时交给精度更高的慢速模型。推理时使用共同的CEM规划流程在模拟器中搜索动作，作者还以真实ALFA飞行数据检查预测。

#### 方法对比分析
与只使用单一RNN隐状态或在线调模型参数的方法相比，AFTER-WM明确将瞬态响应、持续偏差和专家切换分开建模。其优点是可解释并适合短上下文快速辨识；但当前证据主要是低维状态机器人模拟，真实UAV只是离线飞行数据预测，尚未证明图像输入或实体闭环控制。 该框架以模型路由和跨模拟器共享为主要贡献，相比单模型控制可改善报告的模拟回报；模拟器覆盖不等同于实体无人机闭环控制。

#### 实验分析（精简版）
在五种模拟系统和ALFA真实UAV故障飞行数据上，AFTER-WM的多步误差优于十种对照。JSHC模拟控制中使用相同CEM规划器，平均回报93.1，比MOLe高13.8；该控制结论不来自实机。补充材料跨1,586次模拟持久变化检验记忆分工。 在JSHC共同CEM设置下平均模拟回报达到93.1，较MOLe提升13.8；论文统计1,586种模拟变化。真实ALFA仅用于飞行预测评估，未报告实体闭环控制成功率。

#### 实用指南
复现需保证专家、对照模型训练预算和测试窗口一致。作者使用64步上下文、40步预测，快状态维数32、衰减系数ρ=.8；主要模拟结果采用5个随机种子，ALFA为4个。迁移到真实机器人需在线安全约束、视觉状态估计和闭环验证；预测改善不自动意味着策略更安全。 复现要统一CEM候选数和时域，同时核对模拟器路由的触发条件；训练域与推理域的物理参数范围应分别列出。

#### 总结
核心思想：误差分给快慢记忆。
1. 专家根据当前状态和动作预测下一步。
2. 误差更新快速校正、慢速状态和专家路由。
3. 开环预测中快校正衰减，慢记忆持续修正。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.15215v1)
- [arXiv](https://arxiv.org/abs/2611.15215v1)

---

<a id='2611.15048v1'></a>
## [Does Your World Model Know Actions Have Consequences?](https://arxiv.org/abs/2611.15048v1)

**Authors:** Ayush Gaggar, Jared Berry, Todd D. Murphey

**Published:** 2026-11-11

**Categories:** cs.RO

**Abstract:**

Embodied world models (WMs) are implicit learning systems that predict the next future latent state directly from observations and actions. Unique from other end-to-end learning systems, the dynamics predictor of WMs can simulate the environment's response to alternative actions without physical deployment nor pre-specifying states of the dynamical system. Current metrics focus on the quality of model outputs, e.g., clarity of reconstructions or task success, but fail to assess the fidelity of modeling consequences of counterfactual actions. Inspecting the forced response to action perturbations through sensitivity analysis is one way to audit how well WMs can predict the consequences of counterfactual actions. Deploying WMs on hardware will require new evaluation frameworks to analyze the tradeoffs between state error and accurate sensitivity prediction.

### 论文解读
#### 摘要翻译
作者提出动作扰动训练：在学习环境模型时主动注入结构化动作变化，减少策略过拟合于专家轨迹，并改善两项模拟操控任务中的控制效果。 这项工作提供的是可复核的研究结果与方法边界；理解其贡献时应同时看任务设置、观测来源及评价对象。

#### 方法动机分析
模仿或模型学习若只看到窄分布的演示动作，策略在规划时遇到未覆盖动作就可能预测失真。作者关注如何用有限交互扩大动作支持范围，同时让学到的动力学服务于模型预测控制。 关键假设在于现有评估或建模流程遗漏了任务中的重要变量，因此作者把该变量显式纳入流程，并用针对性实验验证。

#### 方法设计详解
训练集用于拟合或构建论文所需组件，推理时按同一任务协议生成预测；训练和推理设置应分开理解。方法将动作扰动融入环境模型训练流程，围绕专家轨迹生成局部动作变化，并比较扰动模型与普通监督模型。在模型预测控制中，规划器利用训练好的动力学模型向前模拟候选动作序列，按任务回报挑选动作。论文还使用约五万条 TD-MPC2 轨迹和总计约十万次环境交互，分析数据量与扰动影响。 从输入到输出的完整流程需要保持各模块接口一致；论文的设计将数据处理、核心预测与评价连接起来，使贡献能够通过消融或对照定位。

#### 方法对比分析
与仅行为克隆相比，动作扰动让模型见到专家轨迹邻域内的变化；与单纯增加随机数据相比，扰动围绕任务相关动作生成。收益依赖仿真环境的精确重置和动力学，且当前仅测 MetaWorld 两个任务，尚不能说明真实机器人或更复杂场景的规划提升。 因而读者应把它看作特定设置下的改进，比较时需要保持数据、指标和预算一致，避免从单一总体指标外推到未测试场景。

#### 实验分析（精简版）
在 MetaWorld push 与 peg insertion 上，论文报告相对模仿学习约16%与48%的性能改善，具体幅度随任务和实验配置而变。实验为模拟任务，且规划效用和扰动策略仍应在统一交互预算、不同随机种子下继续对照。 这些数值描述论文所测范围，评价协议和样本构成会影响绝对分数；作者结果支持方法在本设置下有效，尚需独立数据复验。

#### 实用指南
复现时要固定专家轨迹、扰动分布、训练预算和 MPC 搜索参数，并单独报告模型误差及任务成功率。将方法迁移到真实机器人，先在受控仿真中检查扰动是否落在可行动作范围，再用安全限幅和低风险任务验证。 实施时建议先复核数据许可与预处理，再用小规模样例确认输入输出、随机性和指标实现；保留逐项结果有利于发现迁移退化。

#### 总结
核心思想：在专家动作邻域训练动力学模型。
1. 收集任务相关示范与交互。
2. 对动作施加可控扰动并训练预测模型。
3. 用模型规划并在新状态评估。 结论是有边界的经验贡献，应结合具体需求决定是否复现。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.15048v1)
- [arXiv](https://arxiv.org/abs/2611.15048v1)

---

<a id='2611.14953v1'></a>
## [RSLiveBench-UAV: A Continually Evolving Vision-Language Benchmark for Aerial Spatial Intelligence](https://arxiv.org/abs/2611.14953v1)

**Authors:** Yue Zhou, Jiawei Liu, Jue Chen, Ziyang Gong, Haojia Wei, Jingxiang Li, Xue Jiang, Zhihang Zhong, Pai Peng, Xue Yang, Yansheng Li, Jonathan Li

**Published:** 2026-11-11

**Categories:** cs.CV

**Abstract:**

Aerial spatial intelligence (ASI) requires models to understand objects and spatial relationships across changing viewpoints and scales. Evaluating these abilities as models evolve calls for fresh physical observations, yet producing new aerial spatial questions requires repeatedly establishing entity identities, geometric references, and cross-view correspondences. We introduce \textbf{RSLiveBench-UAV}, a continually evolving vision-language benchmark driven by recurring acquisition of real UAV imagery. Its update protocol admits images captured after the preceding release and delays answer disclosure. An entity-centered production workflow uses geometry-assisted association to link repeated observations of physical vehicles, reuse human-verified attributes, and combine them with view-specific annotations and spatial references for traceable question generation. All tasks use deterministic scoring without model judges. We evaluate 30 general-purpose, remote-sensing, and embodied models on 35,491 question records from ten scenes, covering 17 tasks across 2D perception, 3D perception, cross-view perception, and reasoning. All 18 tested embodied models outperform the three selected remote-sensing comparators in Overall, although the strongest embodied model scores 43.29 versus 59.88 for GPT-6-Astra. Fine-grained vehicle recognition remains challenging, and performance varies across heights and camera pitches. Retrospective scene-group comparisons and rolling-window replay characterize score-pattern consistency under scene renewal. RSLiveBench-UAV connects fresh physical observations, entity-centered answer production, and systematic model evaluation to support continual assessment of ASI. Benchmark will release at https://rslivebench.space

### 论文解读
#### 摘要翻译
RSLiveBench-UAV 把无人机具身问答扩展为跨时间更新的场景理解基准，覆盖17项任务、10个场景和35,491个问题，并比较30个模型。 这项工作提供的是可复核的研究结果与方法边界；理解其贡献时应同时看任务设置、观测来源及评价对象。

#### 方法动机分析
静态图像问答难以检验无人机系统是否持续追踪实体关系、理解场景变化并更新判断。现有评测也常缺少统一任务定义和可复核答案，因此作者构建覆盖多场景的基准及确定性评分流程。 关键假设在于现有评估或建模流程遗漏了任务中的重要变量，因此作者把该变量显式纳入流程，并用针对性实验验证。

#### 方法设计详解
训练集用于拟合或构建论文所需组件，推理时按同一任务协议生成预测；训练和推理设置应分开理解。基准从无人机视频整理问题，包含实体识别与关联、空间关系及时间变化等任务；场景由四个区域更新序列构成，要求模型跨观察保持一致。作者提供标准答案与确定性评分，并区分视觉语言模型和具身模型的能力表现。论文还分析任务类型与场景变化带来的困难。 从输入到输出的完整流程需要保持各模块接口一致；论文的设计将数据处理、核心预测与评价连接起来，使贡献能够通过消融或对照定位。

#### 方法对比分析
相较单帧VQA，基准要求利用连续观察并维护实体关联；相较只报告总体分数，分任务分析可以定位空间、时间及记忆短板。它测量的是对记录视频的回放理解，不直接测试无人机实时控制、主动选视角或在线环境适应。 因而读者应把它看作特定设置下的改进，比较时需要保持数据、指标和预算一致，避免从单一总体指标外推到未测试场景。

#### 实验分析（精简版）
数据集共17项任务、10个场景、35,491个问题，评估30个模型。GPT-6 Astra 得分达到59.88，最强具身对比模型43.29。分数显示当前模型仍存在明显差距，但总体分数受任务构成和标注形式影响，不能直接等同真实飞行能力。 这些数值描述论文所测范围，评价协议和样本构成会影响绝对分数；作者结果支持方法在本设置下有效，尚需独立数据复验。

#### 实用指南
使用该基准时应按场景和任务分别报告结果，检查视频切分、实体同一性标注及重复问题泄漏。它适用于无人机视觉问答和长时序场景理解的离线评估；部署前仍需另做实时延迟、传感器变化、主动观察和飞行安全测试。 实施时建议先复核数据许可与预处理，再用小规模样例确认输入输出、随机性和指标实现；保留逐项结果有利于发现迁移退化。

#### 总结
核心思想：用连续无人机观察测量具身理解。
1. 按任务组织视频问题与答案。
2. 用确定性程序统一评分。
3. 分场景诊断模型的时间与空间短板。 结论是有边界的经验贡献，应结合具体需求决定是否复现。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.14953v1)
- [arXiv](https://arxiv.org/abs/2611.14953v1)

---

<a id='2611.15020v1'></a>
## [Genie Sim Benchmark: Real-to-Sim Fidelity and Evaluation](https://arxiv.org/abs/2611.15020v1)

**Authors:** Di Yang, Da Huang, Yang Lv, Fan Yang, Yi Lu, Zheyuan Xing, Linting Guo, Jiaen Shen, Wenjun Sun, Maoqing Yao, Qian Wang

**Published:** 2026-11-11

**Categories:** cs.RO

**Abstract:**

We evaluate Genie Sim Benchmark through paired real-simulation tasks and complementary system-identification studies. Across 22 robot-task pairs spanning 16 task names, task-specific pi0.5 policies fine-tuned on real-robot data achieve a descriptive Score correlation of 0.9686, MAE 0.0405, RMSE 0.0520, with simulation-real differences from -0.110 to +0.080. G2 crsB experiments assess long-horizon closed-loop position and joint-side torque agreement, while five representative object cases examine identification and cross-engine contact transfer. A separate 32.6503 s teleoperation trajectory, excluded from mechanical identification and servo-gain search, evaluates external position targets through simulated motion-controller command shaping and a calibrated position servo. Across both arms, per-joint position RMSE spans 0.00614-0.03522 rad and current-derived torque RMSE spans 0.73545-4.00099 N m. Native PhysX adaptations recover three recorded interaction tasks, with case-dependent alignment gains. These results support task-specific simulation evaluation through joint assessment of task outcomes and physical-response consistency.

### 论文解读
#### 摘要翻译
Genie Sim 建立真实机器人任务与对应仿真的成对基准，覆盖22组机器人任务配对、16类任务，量化仿真表现与真实执行之间的对应关系。 这项工作提供的是可复核的研究结果与方法边界；理解其贡献时应同时看任务设置、观测来源及评价对象。

#### 方法动机分析
机器人仿真能低成本比较算法，但仿真排名是否能预测真实表现需要直接验证。许多基准缺少同一任务、同一机器人配置下的配对数据，也难区分遥操作误差、状态估计和接触差异。 关键假设在于现有评估或建模流程遗漏了任务中的重要变量，因此作者把该变量显式纳入流程，并用针对性实验验证。

#### 方法设计详解
训练集用于拟合或构建论文所需组件，推理时按同一任务协议生成预测；训练和推理设置应分开理解。作者为每个任务配对记录真实与仿真执行，并定义任务级成功或误差指标；基准包含操作任务、执行轨迹和任务配置。论文另分析遥操作记录精度与由电流推得的力矩信号，作为配对数据质量的诊断，强调真实采集本身也有噪声。 从输入到输出的完整流程需要保持各模块接口一致；论文的设计将数据处理、核心预测与评价连接起来，使贡献能够通过消融或对照定位。

#### 方法对比分析
与单独的仿真榜单相比，成对真实数据允许检查模拟分数能否预测真实结果；其相关性分析仍是基准统计，不代表所有任务或策略都可迁移。遥操作位置误差和电流估算偏差也提醒读者：现实参考不是无误差真值。 因而读者应把它看作特定设置下的改进，比较时需要保持数据、指标和预算一致，避免从单一总体指标外推到未测试场景。

#### 实验分析（精简版）
22组真实—仿真任务配对涵盖16类任务，报告相关系数0.9686、MAE 0.0405。单独遥操作记录约32.65秒，位置RMSE范围0.00614至0.03522，电流推算力矩误差范围0.735至4.001 Nm。任务配对内相关诊断不等同于独立泛化验证。 这些数值描述论文所测范围，评价协议和样本构成会影响绝对分数；作者结果支持方法在本设置下有效，尚需独立数据复验。

#### 实用指南
复用基准时需保持任务定义、机器人与控制器配置一致，明确相关性计算单位，避免把同一任务的多个条件当独立样本。力矩由电流估算，接触研究中的真实回合数有限；报告失败类型并增加不同场地和操作者的复验。 实施时建议先复核数据许可与预处理，再用小规模样例确认输入输出、随机性和指标实现；保留逐项结果有利于发现迁移退化。

#### 总结
核心思想：成对比较真实任务与仿真任务。
1. 建立对应的机器人任务配置。
2. 在真实与仿真中采集一致指标。
3. 分析相关性及测量不确定性。 结论是有边界的经验贡献，应结合具体需求决定是否复现。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.15020v1)
- [arXiv](https://arxiv.org/abs/2611.15020v1)

---

<a id='2611.15429v1'></a>
## [IncR-3DSGG: Incremental Posed RGB Image Based 3D Scene Graph Generation via Privileged Depth Supervision](https://arxiv.org/abs/2611.15429v1)

**Authors:** Mrunmai Phatak, Daniel Kienzle, Roman Küble, Jörg Hähner, Rainer Lienhart

**Published:** 2026-11-11

**Categories:** cs.CV

**Abstract:**

Scene graphs provide a compact, structured representation of environments by capturing objects and their spatial or semantic relationships in a form suitable for embodied agents. Existing 3D scene graph generation methods often rely on depth sensing at inference, either to associate object instances across views via point cloud overlap or to compute inter-object relations from geometric cues, limiting deployment to platforms with depth sensors. We propose IncR-3DSGG, which builds 3D scene graphs incrementally from posed\footnote{RGB images with camera extrinsics estimated by an off-the-shelf visual SLAM system.} RGB images, eliminating the use of depth sensors at inference. Our method leverages a Parametric Relation Predictor (PRP) that learns to estimate pairwise distances and relative angles between objects from 2D visual features and 2D geometric cues, using 3D geometric supervision available only during training. These learned pairwise measurements are fused across views through Pose Graph Optimization (PGO)~\cite{grisetti2010tutorial} to recover metrically grounded 3D object positions and build a globally consistent 3D scene graph incrementally. On the real-world 3RScan~\cite{wald2019rio} dataset, our method attains $52.2$ mean object recall and $30.8$ mean predicate recall under the 3DSSG~\cite{wald2020learning} evaluation protocol, against $27.0$ and $17.6$ for MonoSSG~\cite{wu2023incremental}, while using no depth or reconstructed geometry at inference. This shows that relations and metrically grounded 3D object representations can be predicted from posed RGB images alone.

### 论文解读
#### 摘要翻译
IncR 3DSGG 用深度监督训练、姿态对齐的 RGB 推理，逐步构建室内三维场景图，在3RScan上提升对象与关系召回。 这项工作提供的是可复核的研究结果与方法边界；理解其贡献时应同时看任务设置、观测来源及评价对象。

#### 方法动机分析
场景图既要识别对象和关系，也要在多个相机视角间维持实体身份。逐帧检测容易重复或漏掉对象；许多方法在推理时依赖深度传感器，限制了只提供RGB的应用。 关键假设在于现有评估或建模流程遗漏了任务中的重要变量，因此作者把该变量显式纳入流程，并用针对性实验验证。

#### 方法设计详解
训练集用于拟合或构建论文所需组件，推理时按同一任务协议生成预测；训练和推理设置应分开理解。训练阶段利用深度信息定位对象并监督关系；推理阶段从有相机位姿的RGB序列提取对象与关系，再经 PRP 和 PGO 进行跨视角关联与图优化，维护增量场景图。该分工将深度用于训练信号而不要求测试时深度输入。 从输入到输出的完整流程需要保持各模块接口一致；论文的设计将数据处理、核心预测与评价连接起来，使贡献能够通过消融或对照定位。

#### 方法对比分析
相比依赖深度的基线，方法可在RGB推理时构造关系图并提升对象关联召回；但深度基线的对象位置更精确，说明召回优势并未全面取代几何精度。它假设室内静态场景和已知相机姿态，姿态误差或动态对象可能破坏关联。 因而读者应把它看作特定设置下的改进，比较时需要保持数据、指标和预算一致，避免从单一总体指标外推到未测试场景。

#### 实验分析（精简版）
在3RScan的157个扫描场景上，平均对象召回52.2%、谓词召回30.8%，对比 MonoSSG 为27.0%和17.6%。深度基线对象位置更准，报告距离约0.719–0.764米，而该方法约1.242米。指标揭示召回与定位精度之间的取舍。 这些数值描述论文所测范围，评价协议和样本构成会影响绝对分数；作者结果支持方法在本设置下有效，尚需独立数据复验。

#### 实用指南
复现应记录训练深度来源、相机位姿误差、对象匹配阈值及图优化设置。迁移到移动机器人时，应针对动态物体与遮挡测试实体关联稳定性；若应用依赖精确距离，需结合可靠深度或额外几何校正。 实施时建议先复核数据许可与预处理，再用小规模样例确认输入输出、随机性和指标实现；保留逐项结果有利于发现迁移退化。

#### 总结
核心思想：用姿态对齐的RGB增量维护三维场景图。
1. 训练时用深度监督对象关系。
2. 推理时从RGB抽取实体和关系。
3. 通过PRP与PGO跨视角关联。 结论是有边界的经验贡献，应结合具体需求决定是否复现。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.15429v1)
- [arXiv](https://arxiv.org/abs/2611.15429v1)

---

<a id='2611.15459v1'></a>
## [Fear of Memorizing Objects: End-to-End Zero-Shot Model-Based Object Detection](https://arxiv.org/abs/2611.15459v1)

**Authors:** Leon Greiner, Wout Boerdijk, Junwen Huang, Benjamin Busam, Rudolph Triebel, Maximilian Durner

**Published:** 2026-11-11

**Categories:** cs.CV

**Abstract:**

This paper presents FOMO (Fear of Memorizing Objects), an end-to-end trainable transformer-based architecture for zero-shot model-based object detection. Existing approaches typically follow a proposal-and-match pipeline: target-agnostic mask proposals are generated and then matched to template features in a set of loosely coupled stages, making them susceptible to fragmented proposals in clutter and degraded matching under occlusions. FOMO elegantly replaces this paradigm by injecting object template information early in the detection pipeline: Bidirectional attention between learnable latent tokens, multi-view CAD templates, and query image features produces object-aware image features and conditioned queries which are subsequently decoded into bounding-box and confidence predictions. We train FOMO on Clutter6D, a novel large-scale synthetic dataset with over 500k images featuring more than 10k diverse object models rendered in highly cluttered scenes with multiple instances and varying occlusion levels, which addresses the limited object diversity and lack of realistic dense clutter in existing datasets for training zero-shot multi-instance algorithms. In true zero-shot fashion, FOMO achieves 54.2 mAP on BOP-Classic-Core at 0.23 s per image, ranking second overall with 3.8 pp difference to the top method while being more than 7x faster. Throughout comprehensive ablation studies we investigate and verify architectural design choices.

### 论文解读
#### 摘要翻译
FOMO 用生成式预训练和隐式条件模型实现零样本6D物体姿态检测，在BOP-Core报告54.2 mAP，推理速度约0.23秒。 这项工作提供的是可复核的研究结果与方法边界；理解其贡献时应同时看任务设置、观测来源及评价对象。

#### 方法动机分析
6D姿态估计常依赖每个目标对象专门训练，标注和微调成本高；实际使用还要求及时定位姿态。作者试图将大规模CAD先验转化为可泛化检测能力，同时保留较快推理。 关键假设在于现有评估或建模流程遗漏了任务中的重要变量，因此作者把该变量显式纳入流程，并用针对性实验验证。

#### 方法设计详解
训练集用于拟合或构建论文所需组件，推理时按同一任务协议生成预测；训练和推理设置应分开理解。模型以图像和物体条件信息为输入，ICMT将条件内容编码为潜在token，再预测物体实例的姿态。训练使用Clutter6D的约50万张图像和10,481个CAD模型；论文还在七个彼此分开的真实数据集上微调或适配评估，需区分其零样本主张与后续适配结果。 从输入到输出的完整流程需要保持各模块接口一致；论文的设计将数据处理、核心预测与评价连接起来，使贡献能够通过消融或对照定位。

#### 方法对比分析
相较逐物体专用模型，FOMO强调共享生成先验和无需目标训练的迁移，并以速度换取接近专用系统的精度。它在官方BOP-Core分数低于最高方法，且CAD覆盖、纹理和真实域差异限制零样本适用范围；姿态检测也尚未证明会提升抓取成功率。 因而读者应把它看作特定设置下的改进，比较时需要保持数据、指标和预算一致，避免从单一总体指标外推到未测试场景。

#### 实验分析（精简版）
官方BOP-Core结果为54.2 mAP，推理约0.23秒；相较榜首低3.8分但速度超过七倍。训练集含约50万张Clutter6D图像与10,481个CAD模型。七个真实数据集上的适配实验不能与纯零样本结果混为一谈。 这些数值描述论文所测范围，评价协议和样本构成会影响绝对分数；作者结果支持方法在本设置下有效，尚需独立数据复验。

#### 实用指南
复现需固定BOP评测协议、CAD模型质量、物体对称性处理及推理硬件。评估新域时分开展示纯零样本和适配结果，并加入下游抓取或装配验证；对无纹理、几何相似或未覆盖对象保持谨慎。 实施时建议先复核数据许可与预处理，再用小规模样例确认输入输出、随机性和指标实现；保留逐项结果有利于发现迁移退化。

#### 总结
核心思想：用共享的CAD条件生成先验预测物体姿态。
1. 将图像与物体条件编码为潜在token。
2. 预测候选物体的6D姿态。
3. 在标准基准和目标域分别验证。 结论是有边界的经验贡献，应结合具体需求决定是否复现。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.15459v1)
- [arXiv](https://arxiv.org/abs/2611.15459v1)

---

