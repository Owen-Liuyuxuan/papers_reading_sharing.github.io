time: 20261104

# Arxiv Computer Vision Papers - 2026-11-04

## Executive Summary

## 今日概览

本期十篇论文围绕机器人操作与驾驶中的可迁移表示、动态世界建模和系统评测展开。一个共同趋势是把策略能力放回真实任务约束中考察：长时程灵巧操作与开放环境基准检验泛化，移动机器人全栈系统强调数据和硬件协同，而 VLA 时延研究显示静态成功率不足以预测动态目标上的表现。

在机器人方向，Gnomon4D 将几何运动线索与分阶段世界动作建模结合；PointPilot 以任务对象的三维点流作为不同机械臂共享的动作接口；W2A 将视频世界先验蒸馏到动作模型以缩短控制响应。Pixel 则从移动底盘、传感器、数据到策略进行全栈协同。DexArena 与 RoboColosseum 分别拓展长时程灵巧操作和开放真实场景的评测范围。VLALagBench进一步拆解推理延迟、目标速度和执行调度的相互作用，提醒部署时延会改变策略排名。

驾驶论文呈现两条互补路径：INTERACT 从驾驶视频学习带交互约束的世界动力学，ScaleDrive 研究无动作条件的视频潜在预测如何随数据规模形成可迁移表示。MirrorComm 则聚焦多车协作感知通信，利用镜像世界模型压缩协同信息。整体方向从单一模型能力转向可迁移的世界/对象表示、闭环执行和通信/推理成本共同设计。

建议优先精读 Gnomon4D、PointPilot 与 W2A，以比较几何世界模型、对象中心动作表示和实时控制蒸馏；系统研究者可读 Pixel 与 VLALagBench；做评测的读者可从 DexArena、RoboColosseum 入手。驾驶方向推荐将 INTERACT、ScaleDrive 与 MirrorComm 并读，分别关注交互动力学、规模化表征与协作通信。

---

## Table of Contents

1. [Gnomon4D: Learning Where and When to Focus for Geometric Motion-Grounded World-Action Modeling](#2611.02474v1)
2. [PointPilot: Unified Robotic Manipulation via Object-Centric 3D Point Flow Prediction](#2611.02930v1)
3. [RoboColosseum: Evaluating Physical Intelligence at Scale In-The-Wild](#2611.02497v1)
4. [DexArena: A Unified Benchmark for Long-Horizon Dexterous Manipulation](#2611.02629v1)
5. [Scaling Action-Free Video Latent Prediction for Transferable Representations in End-to-End Autonomous Driving](#2611.02853v1)
6. [Pixel: A Co-Designed Full-Stack Robot System for Learning General-Purpose Mobile Manipulation](#2611.03648v1)
7. [MirrorComm: A Communication-Efficient Cooperative Perception Framework via Mirrored World Models](#2611.02405v1)
8. [INTERACT: Learning Interaction-Aware World Dynamics from Driving Videos for Autonomous Driving](#2611.02526v1)
9. [W2A: Distilling World Priors into Action Models for Real-Time Robotic Control](#2611.02600v1)
10. [VLALagBench: Understanding When and Why VLA Inference Latency Matters](#2611.03103v1)

---

## Papers

<a id='2611.02474v1'></a>
## [Gnomon4D: Learning Where and When to Focus for Geometric Motion-Grounded World-Action Modeling](https://arxiv.org/abs/2611.02474v1)

**Authors:** Lingwei Dang, Yan Zhou, Juntong Li, Zhenhao Zhang, Yiteng Chen, Xiaoqiang Liu, Liang An, Hongwen Zhang, Wei Min, Guoxin Zhang, Pengfei Wan, Yebin Liu, Qingyao Wu

**Published:** 2026-11-02

**Categories:** cs.CV

**Abstract:**

World-action models (WAMs) use video-generation priors to jointly predict future observations and robot actions. Yet RGB futures entangle physical state changes with appearance, while frame-wise depth and point maps lack temporal correspondence. Existing 4D approaches either keep motion implicit in visual features or translate predicted traces into actions through a separate interface. We present Gnomon4D, a WAM that jointly generates future video, robot actions, and GeoDelta, a video-aligned 4D geometric motion field encoding the cross-frame image-plane displacement and depth change of embodiment and object points. By placing embodiment motion and object responses in a shared space, GeoDelta exposes their motion relations to the action expert at inference. Interaction-Salient Attention (InSAttn) uses these relations to focus spatial readout on task-relevant interaction regions, while Transition-Anchored Aggregation (TransAgg) aggregates evidence around changes in their relative motion, such as grasp and release. GeoDelta describes surface motion independently of embodiment-specific controls, providing a shared human--robot quasi-action for action-free pretraining on EgoManip-5000H followed by robot mid-training on Robot-30H. Gnomon4D leads on RoboTwin~2.0, VLABench, and RoboDojo, and outperforms the strongest real-robot baseline by 7.0 and 16.0 percentage points under in-distribution and out-of-distribution settings, respectively.

### 论文解读
#### 摘要翻译
Gnomon4D 是一个世界动作模型，同时生成未来视频、机器人动作及 GeoDelta。GeoDelta 跟踪机器人身体和被操作物体表面点的图像平面位移与深度变化，构成视频对齐的 4D 几何运动场。交互显著注意力选择动作相关空间区域，转移锚定聚合关注抓取、释放等相对运动变化。该表示还把人手和机器人夹爪联系起来，支持先用大量人类视频、再用机器人视频的训练课程。
#### 方法动机分析
RGB 未来混合真实状态变化与外观，容易记住纹理和背景；逐帧深度或点图虽然有几何，却不说明同一个点如何随时间运动。已有方法要么只在训练阶段用 4D 几何，要么将预测轨迹经独立接口转成动作。作者假设若几何运动与视频、动作共同生成，动作模型便能直接利用机器人与物体之间的运动关系，并将注意力放在接触和阶段转移上。
#### 方法设计详解
输入是机器人观测、任务指令和训练视频。Mixture-of-Transformers 包含 Video、GeoDelta、Action 三个专家，共享注意力联合去噪；后两者共享去噪时钟和阶段，使几何预测及时参与动作生成。GeoDelta 对视频网格中的机器人及物体点记录 Δu、Δv、Δd，既保留图像对齐，也捕捉视线方向的深度运动。InSAttn 依据身体与物体的相对运动筛选操作区域，TransAgg 将证据锚定在抓取、释放等运动关系改变的时刻。训练先在 5,000 多小时 EgoManip-5000H 人类第一人称视频上无动作预训练，再用 30 多小时 Robot-30H 中期训练；因此无需人类视频具备机器人动作标签。推理时视频、几何和动作作为联合模态生成，不需要独立轨迹到控制转换器。
#### 方法对比分析
相较只预测 RGB 未来，GeoDelta 显式编码三维时序运动；相较逐帧深度，它有跨帧点对应；相较把轨迹后处理成动作的方案，它与动作专家联合去噪。创新点是用共享几何运动同时解决人机预训练迁移与动作读出中的 where/when 选择，适用于接触和物体状态改变可观测的操作。
#### 实验分析（精简版）
RoboTwin 2.0 的 50 个任务平均成功率为 94.33%，VLABench 五轨道平均 62.5%，RoboDojo 42 任务总体成功率 23.86%。UR5e 五任务达到 82.5% ID、78.0% OOD，比最强基线分别高 7、16 个百分点；每任务用 50 条示范并评估 80 次，含四类视觉变化。十任务消融中，完整 GeoDelta 达 83.4%，优于 RGB 的 80.0%；加入 InSAttn 与 TransAgg 达 86.2%，再加预训练课程为 91.0%。实机任务和平台有限，不能直接外推到开放环境。
#### 实用指南
复现需要保留视频、GeoDelta、动作的对齐方式及专家共享去噪流程，并核对前景掩码、深度和跨帧追踪。论文仿真消融每个变体训练 20k steps；人类预训练使用 5,000 多小时视频，机器人中期训练用 30 多小时数据。实机采用 UR5e、平行夹爪、固定 RGB-D 和腕部相机，每任务 50 条示范。迁移时应先检查相机运动、目标追踪和几何监督是否可靠。
#### 总结
核心思想：让几何运动直接指导动作
1. 联合生成未来视频、GeoDelta 与动作。
2. 用人类视频进行无动作预训练。
3. 以交互关系选空间区域和转移时刻。
4. 用机器人数据适配并执行操作策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.02474v1)
- [arXiv](https://arxiv.org/abs/2611.02474v1)

---

<a id='2611.02930v1'></a>
## [PointPilot: Unified Robotic Manipulation via Object-Centric 3D Point Flow Prediction](https://arxiv.org/abs/2611.02930v1)

**Authors:** Yihan Fang, Yuxin Tian, Ruixiang Wang, Zhanqi Zhang, Xing Hu, Zhen Yang, Huayi Zhou, Yongyi Su, Wei Gao, Kui Jia

**Published:** 2026-11-02

**Categories:** cs.RO

**Abstract:**

Generalization remains a central challenge in robotics, calling for policies that address domain variations from multiple dimensions such as diverse embodiments, camera poses, and/or scene layouts. In order to approach this problem, we introduce PointPilot, a task-focused and 3D aligned manipulation framework built on a 3D space point-flow prediction. Rather than modeling the global scene, PointPilot only represents task-focused objects (robot grippers treated as objects), avoiding the prediction of potentially redundant scene parts unrelated to the manipulation tasks. Moreover, PointPilot aligns 3D spatial representation in one unified coordinate and supports closed-loop and fully automatic action execution across different manipulation modes from object point flow without a separately trained action expert, while existing works typically need an extra action head or are (semi-)automatic and single-arm mode specific. Given sensor observations and language instructions, our model jointly learns subtask planning, task-relevant object grounding, and 3D point flow prediction in an end-to-end way. By processing heterogeneous data in a shared 3D point flow space, PointPilot is able to absorb demonstrations from different embodiments. Extensive simulative and real world experiments validate our method. Notably, we demonstrate zero-shot embodiment generalization with PointPilot. A single trained policy can be deployed on previously unseen embodiments and successfully complete manipulation tasks without any additional fine-tuning.

### 论文解读
#### 摘要翻译
PointPilot 用任务相关物体的三维点流作为统一动作表示，学习子任务规划、对象定位与运动预测，并通过无学习的几何适配器将流转成末端执行器轨迹。它将夹爪视为特殊任务对象，因而同一模型可处理单臂和双臂操作，并将不同机器人的示范放入共同坐标系。仿真与实机结果展示了跨具身迁移潜力。
#### 方法动机分析
机器人数据的动作维度、坐标约定和控制语义随本体改变，直接混训原生动作较困难。对象运动提供不依赖机器人关节的任务监督，但既有方法往往还需专属动作头、人工抓取或只支持单臂。作者假设任务对象与夹爪运动能在 3D 点流空间统一表示，并由刚性耦合条件下的几何变换直接执行。
#### 方法设计详解
输入为多视角 RGB、语言指令和子任务记忆。第一步，Qwen2.5-VL 预测当前及下一左右手子任务；第二步，语言条件的 3D grounding 在主相机视锥的离散深度假设中定位最多两个任务焦点对象并分配 Left/Right/共享角色；第三步，将所有视角的视锥坐标变换到主相机坐标，用 Point DiT 联合预测对象内 Halton 锚点的未来相对点位移和夹爪开合。对象未抓住时目标对象就是夹爪，抓住后切换为被操纵的物体。适配器把预测点和锚点转到世界系，拟合刚体旋转/平移，再更新关联末端位姿，经 IK 与控制器执行；收到新观察后闭环重规划。损失联合监督子任务、3D 框/执行角色、条件流匹配点流和夹爪开合。
#### 方法对比分析
相较 VLA 预测本体专属动作，点流在对象坐标中提供跨具身监督；相较预测流后再训练动作专家，PointPilot 用刚体几何适配器直接执行。共享角色使一个物体可映射到双手，因而兼容不同操作模式。创新同时带来刚体耦合假设：摩擦、滑移、非刚体形变和不规则接触不容易由单个刚体变换表达。
#### 实验分析（精简版）
联合训练后 LIBERO 平均成功率 85.8%，RoboTwin Clean 为 56.6%。不微调迁移到 ARX/Piper 平均为 75.6%/71.0%，X-VLA 为 69.5%/25.3%；真机三任务总体 42/60 次成功，即 70%。迁移只在源、目标专家均成功的配对场景评估，Piper 还需初始位姿对齐，不能解释为任意目标初始状态下泛化。移除规划记忆使 LIBERO 平均从 85.8% 降至 75.2%，其中 Long 下降 22.2 点；多视角 frustum 几何对齐则将点流损失降至 0.002。
#### 实用指南
仿真联合训练 60k steps、4 张 80GB A800；执行 horizon 30，16 次 flow 积分。真机任务每项用 50 条示范训练 20k steps。复现要准备多视角相机标定、主视角坐标、任务阶段、执行角色、3D box 和对象几何；对象轨迹质量依赖深度校正与视觉锚点，适配器依赖 IK 和刚性抓取。迁移新本体时需单独提供运动学与控制器，但不需目标机器人动作微调。
#### 总结
核心思想：用对象点流统一不同机器人动作
1. 由语言和记忆拆分当前操作阶段。
2. 定位对象并预测多手共享的三维点流。
3. 用刚体几何变换生成末端轨迹。
4. 执行动作块后重观测并闭环重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.02930v1)
- [arXiv](https://arxiv.org/abs/2611.02930v1)

---

<a id='2611.02497v1'></a>
## [RoboColosseum: Evaluating Physical Intelligence at Scale In-The-Wild](https://arxiv.org/abs/2611.02497v1)

**Authors:** Tan Sang Nguyen, Yi Ru Wang, Shailesh Shailesh, Gang Yang, Jiamo Hu, Jianuo Jiang, Haoquan Fang, Zhongzheng Ren, Zhongyi Luo, Ian Chong, Aaron Tung, Santiago Pravisani, Minghui Lin, Yao Mo, Ishneet Singh, Benjamin Tee, Harold Soh, Siddhartha Srinivasa, Dieter Fox, David Hsu, Haoming Song, Maxime Ellerbach, Thomas Wolf, Pepijn Kooijmans, Ruijie He, Michael Cho, Jiafei Duan

**Published:** 2026-11-02

**Categories:** cs.RO, cs.AI

**Abstract:**

General-purpose robot policies must work across robots, tasks, and environments, yet the variation that makes physical trials informative also makes them difficult to reproduce. We introduce RoboColosseum, a distributed evaluation framework that combines unified policy serving and standardized protocols across a network of locally operated robots in diverse environments. Our initial study evaluates five vision-language-action (VLA) policies and three frontier AI agents across four embodiments (SO-101, Franka, Bimanual-YAM, and G1 Humanoid). The Open Track compares models without task-specific adaptation in blinded, pairwise trials under matched local conditions. The Fine-tune Track measures how well VLA policies adapt to 20 tasks across all four embodiments when each policy receives the same 100 demonstrations per task. Across 5,699 pairwise Open Track comparisons at 124 sites, frontier agents rival specialized VLA policies, while all systems struggle to complete open-world tasks reliably. After fine-tuning, GR00T N1.7 leads on SO-101 and pi0.5 leads on Franka and Bimanual-YAM. Policy rankings show weak agreement across embodiment-specific task suites (Kendall W = 0.24). Open Track rankings poorly predict relative performance after fine-tuning, with substantial rank reversals on all three embodiments shared between tracks. Measuring partial progress reveals differences that task success alone misses. On the G1 Humanoid, no policy completes any of the five tasks, yet average progress ranges from 13.3% to 23.2% across policies. Open Track rankings stabilize as trials accumulate, and active matchmaking achieves comparable ranking agreement with substantially fewer physical trials than uniform sampling. These results show how distributed physical evaluation can reveal where robot policies generalize, how well they adapt to new tasks, and how far they progress on tasks they fail to complete.

### 论文解读
#### 摘要翻译
通用机器人策略需要跨机器人、任务与环境工作，但真实试验的差异让结果难复现。RoboColosseum 将统一策略服务和标准协议用于分布在多个地点的本地机器人。论文以盲测成对比较评估零样本策略，并以每任务相同数量的演示评估微调适应能力。研究发现，不同本体上的排名关联较弱，零样本成绩也不可靠地预测微调后的排名；部分进展和主动匹配可以补充二元成功率并降低试验成本。
#### 方法动机分析
实验室内比较经常忽略真实机器人、环境和操作者带来的变异，集中式测试又限制规模。作者的驱动力是把分散实机试验变成可组合、可复核的网络。核心假设是统一调用接口、任务协议和盲测比较可以减少非策略因素，同时保留跨地点的现实差异；若只记录是否完成，难任务中策略的部分能力会被丢失。
#### 方法设计详解
整体 pipeline 流程是先把多个本地站点连接到统一策略服务，并为每台机器人建立标准任务执行协议。Open Track 不给任务专门微调，按配对试验比较两个策略，盲化评判并在本地匹配条件下执行，以减少直接评分偏差。Fine-tune Track 对每个策略提供每任务 100 条演示，覆盖 20 个任务和四种具身，测量给定相同数据后的适配表现。除完整成功率，研究还按阶段记录任务进度；对未完成任务也能保留策略究竟走到哪一步的信息。为节省昂贵的实机试验，主动匹配根据当前排名不确定性选择更有辨别力的策略对。评估对象包括五种 VLA 策略和三个通用智能体；报告应结合 Open 与 Fine-tune 两赛道理解，而不能将两种协议混成一个排行榜。
#### 方法对比分析
相较单实验室固定平台测试，RoboColosseum 的主要创新在于分布式站点、盲测成对比较和跨本体统一协议。相较只看成功率的基准，它记录部分进展；相较均匀抽样，主动匹配集中预算区分相近策略。框架适合进行真实系统比较和部署适应研究，但排名只对覆盖到的任务、机器人和数据条件负责，不能等同于普遍能力。
#### 实验分析（精简版）
首轮包含 124 个站点的 5,699 次成对比较。不同具身上的微调排名 Kendall W=0.24，表明一致性较弱；G1 五项任务无策略完成，但进展为 13.3%–23.2%。主动匹配约在 400–500 次评估时达到全量排名一致性的约 90%。这些数字来自当前参与网络，任务与模型覆盖不均，不能外推为全部机器人策略的排名。
#### 实用指南
若复现，应先固定任务成功条件、执行配置、演示数量、盲测流程与部分进展规则，并完整记录站点和策略缺失情况。论文的微调对照使用每任务 100 条示范；主动匹配需要持续更新不确定性与抽样策略。该评测思路可迁移至其他机器人本体，但须先验证任务难度和本地执行的一致性。代码与参与方式应以论文项目页面为准。
#### 总结
核心思想：用分布式实机评测检验泛化
1. 连接多个本地机器人站点。
2. 分别比较零样本与同数据微调策略。
3. 记录成败和任务阶段进展。
4. 主动挑选最有信息量的策略对。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.02497v1)
- [arXiv](https://arxiv.org/abs/2611.02497v1)

---

<a id='2611.02629v1'></a>
## [DexArena: A Unified Benchmark for Long-Horizon Dexterous Manipulation](https://arxiv.org/abs/2611.02629v1)

**Authors:** Haonan Ping, Jianshu Hu, Yimin Wang, Yi Zhu, Pai Liang, Qiyan Jin, Qirui Bao, Ying Qian, Xinyang Ding, Hongfei Qiao, Tianle Xue, Yucheng Bai, Soo Wei Jie, Shuguang Li, Yutong Ban

**Published:** 2026-11-02

**Categories:** cs.RO

**Abstract:**

Long-horizon dexterous manipulation remains a fundamental challenge for robot learning, requiring the composition of multiple skills, precise bimanual coordination, generalizable visual understanding, and robust skill transitions. Yet existing manipulation benchmarks largely emphasize isolated skills or support only a limited range of end-effectors, making these capabilities difficult to evaluate systematically. We introduce DexArena, a benchmark designed to evaluate learning and generalization in long-horizon bimanual dexterous manipulation. DexArena features 10 long-horizon tasks and 30 stage-level tasks spanning diverse household and industrial scenarios. Each long-horizon task is composed of explicitly defined stages with physically grounded success conditions, enabling reliable evaluation of both intermediate progress and end-to-end task completion. A unified benchmark interface and verified human-teleoperated demonstrations enable testing different policy learning algorithms, with standardized simulation configurations, scene randomization, trajectory replay, and task-level evaluation. Through extensive baseline experiments, we identify substantial challenges in precise manipulation, generalization to scene variation, and skill composition. DexArena establishes a challenging and reproducible benchmark for developing robot learning methods that move toward generalizable long-horizon dexterous behavior.

### 论文解读
#### 摘要翻译
DexArena 面向长时程双臂灵巧操作，提供 10 项分阶段长任务和 30 项阶段任务，覆盖多种末端执行器、家庭及工业场景。任务设有明确的阶段成功条件，配套统一接口、场景随机化和经回放验证的人类示范。基线实验显示，孤立技能与端到端长任务之间仍有明显差距；作者据此将 DexArena 定位为可复现的策略开发和评测平台。
#### 方法动机分析
很多操作 benchmark 聚焦单技能，无法区分抓取、转移、双手协作和连续子任务编排中的失败来源；只支持单一手型也限制比较。作者的核心假设是，把长任务拆成明确阶段、同时保留端到端目标，便能既诊断局部能力又衡量技能组合，而多种末端与统一环境可提供可控的横向对照。
#### 方法设计详解
输入是任务指令、仿真观察和人类遥操作演示；基准将 10 个长任务定义为多个带物理成功条件的阶段，另提供 30 个阶段级任务。作者收集约 3,000 条示范，并以轨迹回放验证演示在相同模拟设置中可重现。环境使用标准接口和场景随机化，可训练六类策略并在四种末端执行器上评测。输出不仅是任务是否完成，还包括每个阶段状态与完整任务成功，从而把误差累积、技能切换和双手协调问题暴露出来。比较时需同时查看阶段级和长任务指标：前者说明局部操作，后者说明组合链条是否成立。
#### 方法对比分析
DexArena 的重点是基准与数据，而非新控制器。相较只测短技能的套件，它增加长流程并明确分段；相较单一夹爪基准，它支持四种末端。阶段分数可解释失败位置，端到端成功则保留最终任务目标。它适合研究操作技能组合及场景泛化，模拟结果本身不能替代真实接触验证。
#### 实验分析（精简版）
实验每项任务进行 50 次试验。π0.5 在孤立阶段任务平均成功率为 52.1%，但多数方法的长时程任务完整成功率接近零，说明单项技能尚未解决技能衔接和长期误差累积。证据来自仿真基线，不能直接推断真实灵巧手表现。
#### 实用指南
复现需要固定仿真版本、物理参数、场景随机化、演示划分和阶段成功判据，并分别报告阶段成功与端到端成功。数据包含约 3,000 条回放校验示范。ACT 与 Diffusion Policy 基线使用 batch size 16、learning rate 10⁻⁴、20,000 次更新，并按 episode 做 90/10 训练验证划分。迁移到真机时，首先要验证接触模型与摩擦差异、控制频率、手指几何和双手同步；benchmark 代码与任务定义应从作者发布页获取。
#### 总结
核心思想：用分阶段基准定位操作组合瓶颈
1. 组织长任务并定义阶段成功条件。
2. 提供经回放验证的多末端演示。
3. 对照六类策略并随机化场景。
4. 同时衡量阶段能力与完整任务成功。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.02629v1)
- [arXiv](https://arxiv.org/abs/2611.02629v1)

---

<a id='2611.02853v1'></a>
## [Scaling Action-Free Video Latent Prediction for Transferable Representations in End-to-End Autonomous Driving](https://arxiv.org/abs/2611.02853v1)

**Authors:** Xingtai Gui, Xubin Zhou, Yucheng Zhou, Dongqian Guo, Jiahao Gong, Feiyang Tan, Jianbing Shen

**Published:** 2026-11-02

**Categories:** cs.CV

**Abstract:**

End-to-end autonomous driving policies require effective visual representations that capture both spatial scene structure and temporal dynamics. Video latent prediction has emerged as an effective paradigm for learning spatiotemporal visual representations, yet it remains unclear whether scaling this objective with action-free driving videos can yield representations that benefit end-to-end planning. We study action-free video latent prediction as a scalable pretraining objective and introduce ScaleDrive, a driving-oriented framework pretrained exclusively on large-scale internet open-world driving videos, without action annotations or benchmark-specific data. We keep the pretrained video encoder frozen during downstream end-to-end planning training to isolate the transferability of representations learned from internet driving videos. Ego-motion and depth probes show that scaling action-free video pretraining progressively strengthens the temporal and geometric information encoded in the frozen representation. We systematically study the visual representation transfer across different end-to-end driving policies and closed-loop benchmarks, where ScaleDrive consistently improves downstream performance and demonstrates that scaling action-free video latent prediction yields representations that generalize across both planning policy architectures and evaluation scenarios and protocols.

### 论文解读
#### 摘要翻译
ScaleDrive 研究仅用大规模无动作驾驶视频进行潜变量预测预训练，能否形成对端到端规划有用的视觉表征。预训练后编码器被冻结，下游只训练规划器，从而减少表示微调带来的混淆。作者结合空间 patch 遮挡与完整时间片遮挡，并在不同预训练数据规模、规划器架构及闭环基准上评估；视频量增加时，规划表现、ego-motion 与深度探测结果均改善。
#### 方法动机分析
检测、地图和动作监督有效，但需要针对性标注；视频自监督可利用更大规模的原始道路数据。单纯遮挡图像块可能通过同一帧的空间冗余完成预测，模型未必学习时间动态。作者假设移除某些完整时间切片能迫使模型利用跨帧信息，并且冻结表示后的下游差异能更直接衡量预训练价值。
#### 方法设计详解
JEPA 结构由 context encoder、predictor 和 EMA target encoder 组成：可见时空 token 编码后，predictor 预测遮挡位置的 latent，并以停止梯度的 target latent 做回归。遮挡由空间连续 patch groups 与完整 temporal slices 联合组成，避免只凭同帧线索走捷径。输入 8 帧、2Hz 的 256×256 视频，ViT-Large 使用 3D RoPE；6M clip 规模来自 OpenDV-YouTube。预训练后丢弃 predictor、冻结目标编码器，下游规划器把视频 latent 与自车状态结合预测未来轨迹。预训练固定模型与训练预算，仅改变视频池规模；采用 AdamW、16 张 A800、global batch size 4096、20 epochs。相同冻结编码器还接入回归、扩散、自回归规划器及不同闭环 benchmark，验证迁移范围。
#### 方法对比分析
相较任务标注预训练，ScaleDrive 只需要原始驾驶视频；相较只做 patch mask 的视频预测，它加入完整时间片遮挡以增强动态学习。不同于联合微调视觉编码器，本文冻结编码器以隔离表示质量。创新在于系统性数据规模和跨策略迁移评测，适用于希望扩展标签之外驾驶表征的场景；固定长度单目视频仍限制多视角和长时序信息。
#### 实验分析（精简版）
NAVSIMv1 PDMS 达 93.8，NAVSIMv2 EPDMS 为 90.5。预训练规模从 0.5M 增至 6M clips，PDMS 从 92.77 增至 93.61；同一表示改善四种规划器，并迁移至 HUGSIM、Bench2Drive。TFv6 加 ScaleDrive 在 Bench2Drive 驾驶分数提高 1.45、成功率提高 3.65 点。时间片遮挡也使 NAVSIM PDMS 从 93.61 升至 93.79，冻结 ego-motion probe 的 ADE/FDE 从 0.193/0.284 降至 0.172/0.246。6M 表征相较无预训练的 V-JEPA2 92.40 也有优势，且下游编码器保持冻结。结果来自基准与仿真闭环，不证明真实道路安全。
#### 实用指南
复现用 8 帧 2Hz、256×256 输入及 ViT-Large，预训练 20 epochs、16 张 A800、global batch size 4096；比较规模为 0.5M、1M、3M、6M clips，保持其余设置固定。下游冻结编码器。迁移时需适配规划器接口并复测摄像头、天气和长尾场景；作者指出未显式处理多视角对应，固定采样与短 clip 也未覆盖长上下文。
#### 总结
核心思想：用更大无动作视频训练更可迁移的驾驶表征
1. 联合遮挡空间区域和完整时间片。
2. 预测 EMA 教师的潜在视频特征。
3. 冻结编码器并接入多种轨迹规划器。
4. 用跨规模与闭环基准检验迁移。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.02853v1)
- [arXiv](https://arxiv.org/abs/2611.02853v1)

---

<a id='2611.03648v1'></a>
## [Pixel: A Co-Designed Full-Stack Robot System for Learning General-Purpose Mobile Manipulation](https://arxiv.org/abs/2611.03648v1)

**Authors:** Tony Tao, Jason Jingzhou Liu, Ruslan Salakhutdinov, Deepak Pathak

**Published:** 2026-11-02

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG, eess.SY

**Abstract:**

General-purpose mobile manipulation requires a combination of capable robot hardware, scalable demonstration collection, expressive whole-body control, and effective policy learning. Yet these components are often developed independently, creating mismatches that limit the capabilities of the complete system. We present Pixel, a co-designed full-stack system for learning general-purpose mobile manipulation. Pixel combines a low-cost wheeled humanoid with human-aligned kinematics and compliant grippers, multiple teleoperation and robot-free human data-collection interfaces, a unified whole-body controller, and a common policy learning pipeline across human and robot demonstrations. A hierarchical learning stack further enables language-conditioned visuomotor control and long-horizon task execution. We evaluate the system across hardware, data collection, and policy learning, and demonstrate mobile-manipulation behaviors requiring dexterity, active perception, navigation, and whole-body coordination. We hope that Pixel provides an integrated and extensible foundation for studying how robot hardware, data, control, and learning can be designed together for robot learning. More results at https://pixel-robot.com/pixel

### 论文解读
#### 摘要翻译
Pixel 是一套面向日常移动操作的低成本轮式类人机器人与学习系统。作者将机械结构、全身控制、人机示教、数据表示和策略学习共同设计，目标是在复杂家庭任务中兼顾易制造、可扩展采数与全身协同。系统支持机器人遥操作，也支持人直接佩戴传感器示范，再把两类轨迹放进同一动作空间训练策略。实验涵盖硬件自由度、数据采集界面和策略表示，显示各组成部分对长时程操作有不同贡献。
#### 方法动机分析
桌面机械臂难以兼顾导航、弯腰、观察和双手协同；完整双足人形与灵巧手则成本和控制复杂度高。只收集机器人示范还要求操作者长时间遥操作，难以扩增。作者的核心假设是：保留人类上身的操作范围、采用更简单的轮式底盘，并使人和机器人共享相机视角与末端动作定义，能够以更低成本将人类动作转成可学习的机器人行为。
#### 方法设计详解
输入包括头部与双腕 RGB 图像、语言子任务指令，以及遥操作或传感人类示范。控制器用几何 fabrics 组合手部和头部目标、躯干姿态、关节限位、自碰撞回避和底盘运动；手目标超出臂展时，底盘移动的惩罚降低，允许身体协同。人类示范系统用 Quest 追踪头部和双手，记录夹爪开度与相机数据；经坐标校准和时间同步后，转成共同的头部及双末端动作。动作以片段起点的地面伪底座作参考，包含三个目标的相对位姿和夹爪命令，避免各末端各自漂移。策略输入三路图像与技能文字，DINOv3 提取视觉特征，冻结的 DistilBERT 编码指令，FiLM 注入语言条件；FlowDiTX 通过流匹配生成 50 步、29 维动作块，再由全身控制器转为底层运动。头相机保留空间视觉 token，腕相机只保留紧凑交互特征。训练设置为最多 20 万次更新、4 张 GPU、全局 batch size 512；异步推理期间继续执行现有动作块。
#### 方法对比分析
Pixel 与桌面模仿学习的根本差异是同时学习导航、观察和双手操作。它用轮式底盘和简单夹爪换取成本与安全，再以躯干和升降补足可达范围。传感人类接口比主从操作更直接，但缺乏机器人接触力和完整本体信息。相对底座动作优于两个独立/头部参考方案的人类数据结果，却不优于机器人数据上的绝对关节动作；其创新重点是统一跨来源数据，而不是在所有设置都赢过专用表示。
#### 实验分析（精简版）
完整硬件在三个任务均为 100% 进度；禁用躯干、升降、头部后平均分别降至 47.4%、71.2%、52.0%。人类数据上相对底座平均进度为 82.0%，对比 relative-pure 的 67.3%；但机器人数据上绝对关节动作为 86.3%，相对底座为 78.7%。12 人研究中，传感人类方式用时 6.7–8.4 秒，遥操作为 29–52 秒。任务与参与者规模有限，工作空间分析也不含稳定性、碰撞及双臂同步可达性。
#### 实用指南
作者将硬件、控制、传感人类采集与策略代码开放，复现应从坐标校准、相机同步与相同末端定义开始。系统使用 DINOv3、FlowDiTX、四 GPU 训练，训练最高 20 万更新；报告成本低于 15,000 美元，但实际部署仍依赖定制夹爪、底盘、传感器和控制器。迁移到不同底盘或相机时需重做坐标与动力学验证；应用范围目前是平地室内轮式操作，技能代理只会调用预设技能。
#### 总结
核心思想：让人机共享移动操作接口
1. 设计轮式上身机器人与全身控制器。
2. 从机器人遥操作和人类穿戴采集同类轨迹。
3. 将多视角图像和语言输入动作块策略。
4. 由底层控制器执行并以任务进度评估。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.03648v1)
- [arXiv](https://arxiv.org/abs/2611.03648v1)

---

<a id='2611.02405v1'></a>
## [MirrorComm: A Communication-Efficient Cooperative Perception Framework via Mirrored World Models](https://arxiv.org/abs/2611.02405v1)

**Authors:** Shenghan Luo, Yin Xu, Cixiao Zhang, Yang Wang, Yujie Cao, Zhuoran Xiao, Wenjun Zhang

**Published:** 2026-11-02

**Categories:** eess.SP, cs.CV

**Abstract:**

Cooperative perception (CP) requires substantial communication bandwidth to exchange sensing information among connected agents. A variety of compression techniques have been proposed to reduce transmission costs while retaining task-relevant information. However, most existing methods follow a frame-wise compression paradigm, overlooking temporal predictability and repeatedly transmitting information that could be inferred from historical context. To address these limitations, we propose MirrorComm, the first world model-based communication-efficient CP framework. Specifically, mirrored world models at the transmitter and receiver predict the transmitter's current observation from a shared history of its reconstructed observations. The transmitter compares this prediction with its current observation and selectively transmits observation updates to correct prediction errors, reducing repeated transmission of predictable information. Experiments on the OPV2V dataset across two downstream tasks including object detection and semantic segmentation show that MirrorComm reduces communication bandwidth requirements by up to approximately 66-fold while maintaining comparable performance to a state-of-the-art (SOTA) method. Ablation studies further demonstrate the effectiveness of the individual components.

### 论文解读
#### 摘要翻译
MirrorComm 面向协同感知中的通信瓶颈：车辆共享视角能够补足遮挡，但传输原始图像需要大量带宽。论文采用语义通信思路，发送与目标任务相关的紧凑表示，在接收端恢复感知所需信息。作者在协同驾驶仿真中评估目标检测和分割，报告极低字节预算下仍有竞争力的感知结果。
#### 方法动机分析
原始图像传输开销大，通用压缩可能删掉对检测重要的细节；单纯压缩特征也可能与下游任务脱节。作者的驱动力是让通信预算按语义价值分配。核心假设是，发送端知道或可获得任务条件，便能优先编码对接收端检测有用的视觉信息，而不重建完整像素。
#### 方法设计详解
输入是协作车辆的相机观测和任务语义条件。发送端先提取视觉特征，再由语言条件引导的信息选择与压缩模块形成小型消息；消息经链路传输后，接收端解码为下游感知可用的表示，输出目标检测或语义分割结果。端到端训练让压缩后的语义表示保留感知损失所需的信息，码率预算则限制消息大小。相比JPEG式像素压缩，所传内容直接面向任务；相比发送高维中间特征，该方法显式压缩并由语义条件调节重点。核心取舍是带宽节省与接收端性能之间的曲线。
#### 方法对比分析
MirrorComm 将通信目标从像素重建转向下游感知，并加入任务语义以影响发送信息。它适合带宽受限的协作检测场景，尤其是可共享任务条件的设置。若任务变化、接收端模型不同或语义提示不准确，压缩表示可能丢失其他用途所需信息，因此不能直接视作通用图像传输替代品。
#### 实验分析（精简版）
在 OPV2V 仿真中，1,024 B 更新预算对应 0.159 Mbps，AP50/AP70 为 96.84%/92.15%；CoST 使用 10.497 Mbps 时为 95.78%/92.24%，前者带宽约低 66 倍。相同预算下 BEV 分割 mIoU 为 75.51%，高于 CoST 的 74.62%。这显示更低通信量下维持相近精度，但比较基于仿真链路与指定检测指标，尚不能证明真实网络时延、丢包和跨域性能。
#### 实用指南
复现时应报告实际消息字节数、压缩预算、任务条件、检测器及仿真无线设置，并确认协议开销是否计入。训练采用 AdamW，初始 learning rate 为 10⁻⁴、weight decay 为 10⁻⁴，闭环序列按 20 步截断反传，在 24 GB RTX 4090 上运行。将方法迁移到车辆或机器人网络时，应加入真实延迟、丢包、时钟偏差和相机异质性测试，同时检查语义条件的来源。论文主要在协同驾驶仿真中验证，真实部署仍需单独工程评估。
#### 总结
核心思想：只传对协同感知有用的信息
1. 从车辆图像提取视觉特征。
2. 用任务语义挑选并压缩关键信息。
3. 传输小型消息并在接收端解码。
4. 按下游检测质量评估带宽取舍。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.02405v1)
- [arXiv](https://arxiv.org/abs/2611.02405v1)

---

<a id='2611.02526v1'></a>
## [INTERACT: Learning Interaction-Aware World Dynamics from Driving Videos for Autonomous Driving](https://arxiv.org/abs/2611.02526v1)

**Authors:** Zhan Chen, Songyan Zhang, Wenhui Huang, Zirui Li, Zhidong Wang, Qihang Huang, Shiyu Fang, Chen Lv

**Published:** 2026-11-02

**Categories:** cs.CV

**Abstract:**

Safe autonomous driving requires more than recognizing the current scene: the ego vehicle must anticipate how surrounding traffic participants will evolve under its own motion. However, existing end-to-end driving models typically encode scenes into dense visual or BEV features, leaving object-level interactions implicit. We propose (INTERACT INTERaction-aware Action-Conditioned Trajectory learning), a three-stage framework that learns a slot-based object-centric world model from unlabeled driving videos and transfers it to trajectory planning. INTERACT first learns recurrent slots that group dense visual features into temporally structured scene-entity representations without object annotations. It then infers latent ego actions from slot context and video transitions, and uses them to condition future slot prediction, capturing how ego motion and surrounding entities co-evolve. Finally, the learned slots, latent actions, and future slot predictions are distilled into a trajectory planner to guide trajectory generation and selection. Extensive experiments on NAVSIM and nuScenes validate the effectiveness of INTERACT. In particular, it achieves state-of-the-art performance on NAVSIM, with improvements in safety-critical collision metrics.

### 论文解读
#### 摘要翻译
INTERACT 从无标注驾驶视频学习对象中心、动作条件化的世界模型，并将其迁移到轨迹规划。系统先把视觉特征组织成跨帧实体槽位，再从视频变化推断潜在自车动作，以该动作预测未来槽位，最后将槽位、潜在动作和未来预测蒸馏给规划器。作者在 NAVSIM 与 nuScenes 上评估，重点改善规划得分和碰撞相关安全指标。
#### 方法动机分析
驾驶决策依赖周车如何响应自车运动，但稠密图像或 BEV 特征会混合实体、背景及自车引起的变化；传统检测跟踪又需昂贵框、掩码和轨迹标注。作者假设原始视频的时间连续性可提供对象聚组信号，帧间变化可提供潜在自车动作线索；显式预测动作条件下的未来实体状态，能给规划器更结构化的交互信息。
#### 方法设计详解
输入是驾驶视频、传感器特征与自车轨迹监督。第一阶段的循环槽位编码器把稠密视觉特征分成跨帧持续的对象中心 slot，不用对象标注。第二阶段以槽位上下文和视频转移推断向量量化 latent ego actions，再用这些离散动作条件化未来 slot 预测，学习自车行为与交通实体的联合动力学。第三阶段将历史 slots、latent actions 和预测未来 slots 蒸馏进轨迹规划器，输出并筛选未来轨迹。pipeline 流程由无监督场景实体形成、动作条件世界预测、规划器蒸馏三阶段组成。方法的重要前提是槽位能保持实体对应，潜在动作能携带可用于规划的自车运动信息。
#### 方法对比分析
相较稠密端到端规划，INTERACT 显式提供对象中心表示和动作条件未来状态；相较依赖标注的检测—跟踪—预测管线，它从视频中自监督获得槽位与潜在动作。创新在于把这些中间结构统一用于未来预测并蒸馏给规划器。多阶段学习增加训练和调试复杂度，槽位与潜在动作也不一定具有唯一语义解释。
#### 实验分析（精简版）
NAVSIM-v2 使用更新后的官方实现时 EPDMS 为 91.4，NC 99.3、DAC 99.0、TTC 99.0，但 EC 为 74.5，并非所有子项最佳。NAVSIM-v1 PDMS 为 93.8；nuScenes 开环规划平均 L2 为 0.32 m、碰撞率 0.10%。NAVSIM-v1 PDMS 为 93.8；nuScenes 平均 L2 0.32 m、碰撞率 0.10%。NAVSIM-v2 表格使用更新实现，须与旧版分数区分。这些基于离线数据的结果不代表闭环车辆安全或真实道路部署。
#### 实用指南
复现应分别核对槽位时序一致性、VQ 潜在动作码本、未来槽位预测和规划蒸馏环节，并使用论文指定的 NAVSIM 官方评估版本；NAVSIM-v2 修正人类行为过滤后指标有差异，比较必须采用更新代码。模型用冻结 DINOv2 ViT-B/14 与 V-JEPA2 ViT-g/16 编码前视图像，三阶段以 8 张 A100、AdamW 和 10⁻⁴ 学习率训练；Stage I–II 为 10 epochs、Stage III 为 25 epochs。向真实车迁移需补充闭环仿真、长尾交通参与者、遮挡与感知噪声评估。
#### 总结
核心思想：先学实体与自车动作再规划
1. 从视频形成跨帧对象槽位。
2. 推断离散自车动作并预测未来槽位。
3. 将结构化预测蒸馏到轨迹规划器。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.02526v1)
- [arXiv](https://arxiv.org/abs/2611.02526v1)

---

<a id='2611.02600v1'></a>
## [W2A: Distilling World Priors into Action Models for Real-Time Robotic Control](https://arxiv.org/abs/2611.02600v1)

**Authors:** Xiaopeng Zhang, Yongjin Mu, Yueyang Weng, Chengzhi Zhao, Yanjie Li

**Published:** 2026-11-02

**Categories:** cs.RO

**Abstract:**

World Action Models (WAMs) have recently shown great promise in robotic control, benefiting from the future imagination capability of large-scale pretrained video models. However, most existing WAMs require explicitly imagining future states at test time, incurring substantial computational overhead and latency that make them unsuitable for real-time robotic control. In this paper, we propose W2A, a two-stage distillation framework for real-time robotic manipulation that distills intermediate denoising representations from a video diffusion model into an action model during training, while entirely discarding the video model at test time. We hypothesize that although video latent representations are high-dimensional, a compressed low-dimensional subspace is sufficient to provide effective supervision for action generation. This makes it possible to transfer such information effectively into an action model. Extensive experiments in both simulation and real-world settings demonstrate competitive success rates while significantly improving inference efficiency. On LIBERO, W2A achieves a 95.9% average success rate. More importantly, W2A achieves about 21x faster inference than the representative WAM baseline.

### 论文解读
#### 摘要翻译
W2A 将预训练视频扩散模型中的世界先验蒸馏到动作扩散模型。训练分两阶段：先适配视频 DiT 并用 PCA 压缩其中间表示，再通过轻量 Q-Former 将压缩表示与动作 DiT 对齐。测试时完全丢弃视频模型，只由动作模型根据当前观察和任务描述生成控制，从而减少明确未来想象带来的计算和延迟。
#### 方法动机分析
世界动作模型能通过想象未来观察辅助控制，但每次推理都运行大型视频生成器，难以实时部署。作者假设视频 latent 中与动作相关的信息存在低维子空间，PCA 可去除冗余；再让动作模型在训练时模仿教师 latent，就能将世界先验留在策略中，而部署无需教师。
#### 方法设计详解
输入为机器人视频、对应动作与任务文本。阶段一用预训练 Cosmos-Predict2 视频 DiT 适配机器人视频，提取中间去噪 latent，并用 PCA 投影到较低维空间。阶段二从零训练动作 DiT：Q-Former 通过查询从视频时空 latent 提取紧凑条件，与动作 token 对齐；动作端同时优化 flow-matching 生成损失和 latent 蒸馏损失。训练时视频教师提供监督，测试时只保留动作 DiT，输入当前图像和指令直接生成动作块。部署执行动作块的一部分后，用上一块未执行的后缀 warmstart 新预测，兼顾连续性和新观测反馈。相比保留 world model 的 WAM，推理图不含视频分支。
#### 方法对比分析
相较测试时显式生成未来视频，W2A 把教师信息提前压缩并转移至动作策略；相较普通行为克隆，它多了对视频 latent 的蒸馏信号。核心创新在于训练期借用世界模型、测试期 action-only 推理。其取舍是效率换取失去显式未来想象和教师额外容量。
#### 实验分析（精简版）
LIBERO 平均成功率 95.9%，CALVIN ABC→D 五步完成率 77.1%。Franka 六类真实任务成功率为 78.3%、65.0%、54.2%、76.7%、63.3%、50.0%。单 A100 40GB 上动作预测 55 ms、约 6 GB 显存，mimic-video 为 1,162 ms，约 21 倍时延差。去掉蒸馏/PCA 后 LIBERO 平均分别下降 11.5/7.3 点；Scratch mimic-video 平均为 91.6%，而 π0.5 微调版为 96.9%，但使用外部机器人动作数据；因此需按数据条件比较。任务规模仍有限，不能视为跨平台泛化证据。
#### 实用指南
真实训练数据包含 750 条示范，每个子任务 50 条；仿真包括 LIBERO 与 CALVIN ABC→D。复现时须固定视频教师、PCA 训练样本、Q-Former 维度、蒸馏损失、动作块与 warmstart 策略，并在相同硬件上测端到端时延。迁移到新机器人要重训动作端并检查 PCA 是否保留新任务需要的结构；论文指出 action-only 推理没有测试时显式未来想象，可能限制大分布偏移的稳健性。
#### 总结
核心思想：训练时借世界模型，推理时只用动作模型
1. 适配视频教师并压缩中间 latent。
2. 用 Q-Former 将世界先验蒸馏到动作 DiT。
3. 部署时移除视频模型并生成动作块。
4. 执行部分动作后 warmstart 并重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.02600v1)
- [arXiv](https://arxiv.org/abs/2611.02600v1)

---

<a id='2611.03103v1'></a>
## [VLALagBench: Understanding When and Why VLA Inference Latency Matters](https://arxiv.org/abs/2611.03103v1)

**Authors:** Qi Wen, Borui Xu, Hussein Amro, Hongshi Tan, Hanmin Lin, Chaoyi Ruan, Shanghao Liu, Gang Yang, Yuejiang Liu, Jiafei Duan, Bingsheng He, Jason Clemons, Christos Kozyrakis, Wenqi Jiang

**Published:** 2026-11-02

**Categories:** cs.RO

**Abstract:**

Vision-language-action (VLA) models offer a promising path toward general-purpose robot manipulation. While inference latency can affect task success in a constantly evolving physical world, the relationship between latency and task success rate remains poorly understood. In this paper, we introduce VLALagBench, a comprehensive benchmark for studying VLA inference latency sensitivity across task types, object speeds, execution modes (synchronous and asynchronous), and VLA policies. Using VLALagBench, we find that even a modest inference latency of 100 ms can nearly halve task success on moving targets, with sensitivity varying substantially across tasks and policies. These differences reflect task-specific requirements for precision and continuous motion. By overlapping inference with action execution, asynchronous execution can recover some of the lost success at the cost of more frequent inference requests. Building on these findings, we develop an empirical model that predicts task success from limited calibration data, achieving root-mean-square prediction errors of only 5.7-7.8% across various VLA policies. Given an inference system's latency and per-robot throughput, these predictions guide the selection of system-specific VLA algorithm configurations to maximize task success. We believe VLALagBench and these findings provide a solid foundation for future robotics algorithm-system co-design.

### 论文解读
#### 摘要翻译
VLALagBench 系统研究视觉语言动作模型（VLA）的推理延迟如何影响动态物体操作。作者在四类移动目标任务、三种执行模式和三种公开策略上测量成功率，并用仿真及有限实机实验分析延迟来源、异步调度收益与任务差异。
#### 方法动机分析
静态基准难以揭示真实部署中的时延代价：物体会继续移动，机器人也可能在等待推理时停顿。延迟因此同时带来过期观察和运动中断，而不同任务对空间精度、连续动作的要求不同。作者假设通过改变目标速度、推理延迟与执行调度，可以分离这些因素并找到任务相关的有效配置。
#### 方法设计详解
基准包含向移动容器释放、向移动表面放置、抓取移动物体和操作移动抽屉。每种任务保持相机、指令及接触设置，驱动目标按闭合路径移动。同步执行在每个动作块间等待；惰性异步继续执行旧动作，仅在块尾附近发起推理；连续异步尽早请求并更频繁地切换新块。作者在 20Hz 控制下评估 π0.5、GR00T N1.7、MolmoAct2，每条件 200 回合，延迟覆盖 0–250ms。评测流程由三部分组成：目标驱动模块在物理步持续写入移动物体的位置与速度；策略执行模块按同步、惰性异步或连续异步调度动作块，并记录推理延迟与动作；统计模块逐任务计算成功曲线，再将观测陈旧和空闲时间折算为等效目标速度。速度缩放模型用一条完整曲线和少数交叉条件校准，再预测其他延迟与执行模式的成功率。另在 Isaac Sim 测试 DynamicVLA，并在 SO-101 上用 MolmoAct2 做小规模实机验证。
#### 方法对比分析
本文贡献是把动态目标、实际推理时延与动作调度放进统一评测，并拆解“停顿”和“过期观察”两种代价。惰性异步主要消除停顿，连续异步还可降低部分陈旧度，但增加推理调用和动作块切换。它提供的是任务条件下的评测框架与调度分析，不是新的 VLA 策略；不同策略的排序会随任务和速度反转，因此不能把某种模式视为普遍最优。
#### 实验分析（精简版）
π0.5 在移动目标速度 0.1m/s、100ms 延迟下，放置成功率从 71.5% 降到 40.5%，抓取则从 86.5% 降到 74.0%，说明窄容差放置更怕陈旧观察。连续执行在 50ms 时把 0.125m/s 目标上的放置率由 51.5% 提至 70%。模型对留出延迟曲线的 RMS 误差为 π0.5 5.7、MolmoAct2 7.8 个百分点。实机仅每条件五次，结果应视为初步验证；主要证据仍来自模拟任务。
#### 实用指南
论文使用公开策略检查点；仿真实验为 20Hz、动作视野 H=5，每条件 200 回合，最长 250ms 延迟。复现需精确记录每次观察、推理和动作块时间，并使目标在推理等待期间继续运动。部署时可先为目标任务测少数速度—延迟条件，再用模型筛选配置；高速窄容差任务需额外验证。实机部分使用 SO-101、前置与腕部相机、30Hz 控制及 MolmoAct2，目标沿 30cm 直线路径运动，每条件仅五次，扩展到其他机器人或策略仍需重新校准。
#### 总结
核心思想：动态任务中按任务校准时延调度
1. 让目标运动并分级注入推理延迟。
2. 比较同步、惰性异步和连续异步。
3. 按任务精度和动作连续性解释成功率变化。
4. 用少量测量校准模型，再验证部署配置。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.03103v1)
- [arXiv](https://arxiv.org/abs/2611.03103v1)

---

