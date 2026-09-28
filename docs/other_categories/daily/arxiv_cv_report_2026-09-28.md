time: 20260928

# Arxiv Computer Vision Papers - 2026-09-28

## Table of Contents

1. [VisTacAlign: Co-Training Dexterous Policies on Tactile Human and Robot Demonstrations](#2609.30959v1)
2. [TRACKGRAPH: Online Open-Vocabulary 3D Scene Graphs via Image-Space Tracking](#2609.31005v1)
3. [Enabling a Unified Cross-Domain Representation for Two-Finger Gripper Manipulation via Interaction-Centric Modeling](#2609.31207v1)
4. [Imp-ACT: Adaptive Impedance Control and Action Chunking with Transformers to Learn Contact-Rich Manipulation from Demonstrations](#2609.31225v1)
5. [Quadruped Obstacle Avoidance and Footstep Planning with Distributed Low-cost Time-of-Flight Sensors](#2609.31008v1)
6. [PHASE: Compliance-Enabled Tactile Phase Retrieval for Few-Shot Insertion Learning](#2609.30889v1)
7. [TACTIC: Understanding Tactile Encoders and Conditioning for Contact-rich Robot Manipulation Policies](#2609.30969v1)
8. [DualManip: Agentic Dynamic Manipulation via Dual-Path Semantic Reasoning and Geometric Adaptation](#2609.31112v1)
9. [Bundled Contact Gradients: Stabilizing Differentiable Simulation for Deployable Dynamic Tasks](#2609.30951v1)
10. [Transformer-based Monte Carlo Localization in Construction Meshes](#2609.31357v1)

---

## Papers

<a id='2609.30959v1'></a>
## [VisTacAlign: Co-Training Dexterous Policies on Tactile Human and Robot Demonstrations](https://arxiv.org/abs/2609.30959v1)

**Authors:** Julien Poffet, Matthew Strong, Ankush Dhawan, Baiyu Shi, Shalika Neelaveni, Yujia Yuan, Zhenan Bao, Monroe Kennedy

**Published:** 2026-09-25

**Categories:** cs.RO

**Abstract:**

Human demonstrations are a cheap source of data for dexterous manipulation, but co-training a robot policy on them requires closing the human--robot gap in every modality the policy consumes. We present VisTacAlign, a framework for co-training 3D-visual-tactile dexterous policies on human and robot demonstrations. Glove-tracked human hand motion is retargeted to a 17-DoF tactile robot hand with a one-time fingertip correction. The human hand is then erased from both stereo views and replaced by a posed robot-hand mesh painted with pixels from robot recordings, and a real-time stereo foundation model is re-run on the composite, so the human point clouds carry the same stereo errors and visibility as the robot ones. Finally, a capacitive tactile glove is aligned to the robot's fingertip sensors in its signal space, giving one interpretable per-finger force representation. A diffusion transformer consumes point-cloud, proprioceptive, and per-finger tactile tokens. On three real-world tasks requiring precise force -- Lego assembly, plucking strawberries of varying size, and activating and lifting a power drill -- adding aligned human demonstrations to existing robot data improves over robot-only policies, and ablations show that both tactile input and visual alignment are necessary. Project page: https://vis-tac-align.github.io

### 论文解读

#### 摘要翻译
人类示范是灵巧操作的低成本数据，但共同训练仍须弥合策略所用模态的人机差异。VisTacAlign 将手套追踪动作重定向至 17 自由度触觉手并校正指尖；擦除双目图中的人手，换入以机器人记录像素着色的手网格，再次运行立体模型，使人类点云带有类似机器人观测的立体误差与可见性；把电容手套信号映射到机器人指尖传感器空间。扩散 Transformer 融合点云、本体感觉和逐指触觉。乐高装配、草莓采摘和电钻操作的真实实验显示协同训练优于机器人单独训练，消融支持触觉及视觉对齐的作用。

#### 方法动机分析
单纯把人手动作映射成机器人关节，仍留下外观、立体测量和触觉量程的域差异：手形误差会破坏捏持，遮挡与颜色影响点云，电容信号也不等于机器人力传感器单位。论文假设应直接对齐策略消费的视觉、运动与触觉表示；人类示范采集较快，但触觉映射仍需要同任务机器人接触数据。

#### 方法设计详解
动作侧用 Apple Vision Pro 指尖位置作伪真值，训练小型 MLP 修正手套关节旋转，单关节修正不超过 20°；指尖中位误差由 13.7 降至 7.1 mm，拇指—食指捏持误差由 11.8 降至 3.4 mm。再以可微前向运动学和 RMSprop 匹配掌心—指尖、拇指—指尖及相邻指 PIP 向量，求机器人关节角；腕部 tracker 到工具中心点以标记板标定，均值残差低于 5 mm。触觉侧将机器人每指正向 taxel 力求和；人类电容读数经 log(1+x) 变换后逐指求和。每项任务用熵正则最优传输重心映射，把人类五维力向量对齐到机器人力分布，ε=0.3；无接触保持零，机器人数据不变。视觉侧用 SAM 3 分割物体并遮去人手，将机器人手网格按重定向姿态放回图像、以机器人像素着色，再对合成双目图运行 Fast-FoundationStereo。策略输入两个时刻各 4096 个 XYZ+RGB 点，经 PointNet 编为 128 维 token；另含末端位姿、17 个关节角和五指触觉 token。23.5M 参数、六层 DiT 用一致性流匹配预测 64 步动作（2.1 秒）；batch size 64、学习率 1e-4、推理 10 个流步骤，每次执行 32 步后重规划。

#### 方法对比分析
区别于只做运动重定向或视觉共训，VisTacAlign 同时对齐动作、双目测量过程和触觉信号。与直接把理想网格点拼入点云相比，它将机器人手投回双目图再重跑立体模型；与学习隐空间触觉相比，任务内最优传输保留可解释的逐指机器人力单位且不改机器人数据，但需机器人接触样本，并丢失剪切力及 taxel 布局。方法适用于短时、准静态且依赖精确施力的灵巧任务。

#### 实验分析（精简版）
电钻任务中，26R 点云策略成功率为 50%，加入 26 个人类示范及对齐触觉后升至 70%。草莓小、中、大尺寸的成功率从无触觉机器人基线的 60%、70%、0%，升至协同训练的 90%、80%、70%，各尺寸测 10 次。乐高任务中，30R 完成率 10%，50H30R 且视觉、触觉均对齐时为 80%；去除视觉对齐后，50H10R 从 40% 降至 10%，去除触觉后 50H30R 为 70%。结果支持对齐作用，但试验次数不大且无置信区间；草莓比较同时改变人类数据和触觉输入，不能单独区分二者贡献。

#### 实用指南
论文报告手套 60 Hz、机器人触觉 30 Hz、推理 10 步；每个策略用单张 RTX 4090 训练 5–25 GPU 小时，RTX 4080 上重规划需 269 ms。复现需要双目相机、腕部—末端标定、机器人手运动学、逐任务人机接触力样本及 SAM 3 分割；部分优化器/网络超参数与数据规模未充分说明。论文给出项目主页，但未明确代码、模型或数据的公开状态。迁移需重做运动学、手部外观合成与触觉映射。

#### 总结
核心思想：对齐人机三模态后共训。
1. 校正手指并重定向到机器人关节。
2. 把人类逐指触觉映射到机器人力单位。
3. 将机器人手合成进人类双目图并重算点云。
4. 用视觉、本体感觉和触觉共同预测动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.30959v1)
- [arXiv](https://arxiv.org/abs/2609.30959v1)

---

<a id='2609.31005v1'></a>
## [TRACKGRAPH: Online Open-Vocabulary 3D Scene Graphs via Image-Space Tracking](https://arxiv.org/abs/2609.31005v1)

**Authors:** Peder Borge Hellesylt, Albert Gassol Puigjaner, Kostas Alexis, Annette Stahl

**Published:** 2026-09-25

**Categories:** cs.CV, cs.RO

**Abstract:**

Open-vocabulary 3D maps enable robots to reason about previously unknown environments using natural language. However, existing systems typically segment every incoming image, associate detections with persistent 3D segments, and frequently perform costly Vision-Language (VL) inference. We present TRACKGRAPH, an online open-vocabulary system that maintains short-term 2D mask identity directly in the image stream before fusing segments into 3D. FastSAM masks and CLIP features are computed at sparse keyframes, while dense DINOv3 features are used to propagate masks at a high rate in between. The resulting tracked masks are fused into a class-agnostic 3D segment layer within a hierarchical scene graph, with 3D association handling tracking interruptions and long-term revisits. Compact multi-view CLIP embeddings enable open-vocabulary retrieval. Across Replica, ScanNet++, and HM3D, TRACKGRAPH achieves competitive open-vocabulary segmentation and retrieval against state-of-the-art mapping methods, including the highest synonym frequency on Replica (0.50). On the same NVIDIA A100, it is 1.7x faster and uses 3.3x less GPU memory than ViT-H OVI-MAP. Real-world quadruped deployments demonstrate onboard scene graph construction and object search at 7.5Hz, while recorded drone data is used to test the method under aerial viewpoints.

### 论文解读

#### 摘要翻译
开放词汇三维地图能帮助机器人用自然语言理解陌生环境，但逐帧分割、跨视角关联和视觉—语言推理代价高。TRACKGRAPH 先在图像流中维护短期掩码身份：关键帧运行 FastSAM 并提取 CLIP 特征，中间帧借助稠密 DINOv3 特征传播掩码，再把跟踪结果融合进层级场景图的类别无关三维片段层；三维匹配处理跟踪中断和重访，多视角 CLIP 特征支持开放词汇检索。作者在 Replica、ScanNet++、HM3D 上报告有竞争力的分割与检索结果；Replica 同义词频率达 0.50。同一 A100 上比 ViT-H OVI-MAP 快 1.7 倍、少用 3.3 倍 GPU 显存；四足机器人机载演示以 7.5Hz 建图并搜索，无人机录制数据用于检验空中视角。

#### 方法动机分析
以往方法通常先对每帧分割，再依赖三维重建或渲染重叠关联对象，VL 特征也可能频繁计算。作者的关键假设是：相邻图像中的 DINOv3 局部特征足以短时保持掩码身份，因此连续性可先在二维解决，三维主要负责断轨合并和离开活动区域后的重访。这减少重复推理，但性能仍依赖深度、位姿及图像跟踪；在大型多样场景里，维持对象完整覆盖较困难。

#### 方法设计详解
输入是 RGB-D 序列与里程计。每帧用 DINOv3 ViT-S+ 提取稠密特征；每隔 24 个处理帧，FastSAM-x 产生类别无关掩码，OpenCLIP ViT-H/14 异步提取掩码区域特征。关键帧检测与既有轨迹按 IoU 和 DINOv3 特征相似度匹配，未匹配检测新建 source track。中间帧对局部相似 patch 的轨迹概率作 softmax 加权传播，取像素级最大概率得到掩码。随后按位姿把轨迹掩码融合到 TSDF，每个体素保留最多 4 个带权轨迹假设；关键帧权重为 1，传播结果依时间衰减，减轻粗糙边界和旧掩码的影响。网格顶点由最高权重轨迹归属，形成类别无关片段；活动窗口内依据共享顶点、双向重叠及 DINOv3 相似度合并断裂轨迹。历史片段先用质心 kd-tree 找近邻，再综合表面重叠与外观相似度决定是否重访合并。每条轨迹保留独立 CLIP 视角特征，文本或图像查询取与各视角的最大余弦相似度作为片段分数。

#### 方法对比分析
ConceptGraphs、HOV-SG、OVI-MAP 等从逐帧掩码出发做三维关联；FindAnything 借助当前渲染区域匹配。TRACKGRAPH 的区别是把短期身份跟踪前移到图像空间，以三维关联处理较少发生的断轨与长期重访；CLIP 也按轨迹而非逐帧或逐点存储，并保留多视角而不平均。它适合连续 RGB-D 机器人视频和在线场景图构建，前提是位姿及深度可靠；不保证复杂大场景中的片段总能覆盖完整物体。

#### 实验分析（精简版）
OpenLex3D 分割评估中，Replica 同义词频率为 0.50，高于 HOV-SG 的 0.45；ScanNet++ 为 0.35，低于 HOV-SG 的 0.40。Replica 检索 AP50 达 15.78%，高于 ViT-H OVI-MAP 的 14.21%，但 AP 仍落后于 OVI-MAP。Replica 跟踪消融中，DINOv3 传播将无丢帧输入率从 FastSAM+BoT-SORT 的 6Hz 提至 15Hz，AP 基本相当（6.00 对 6.01），AP50 则为 15.78 对 13.74。A100 上 3cm 配置运行 8.9 分钟、显存 8.3GB；ViT-H OVI-MAP 为 15.1 分钟、27.2GB，不过后者每十帧处理一次，前者处理每帧，速度比较需谨慎。HM3D 检索 AP/AP50 较弱，可能反映片段未覆盖完整物体。真实四足机器人三处环境报告所有查询目标被检索到，但未提供目标数或统计区间；无人机实验基于录制数据。

#### 实用指南
论文给出的基准设定包括 DINOv3 ViT-S+（29M 参数，图像短边 720）、每 24 帧运行关键帧模型、3cm TSDF；真实部署改用 CLIP ViT-B、短边 480、0.125m TSDF，并以 30Hz 相机每四帧处理一次，标称 7.5Hz。关键帧关联阈值及传播衰减系数应结合机器人、场景尺度和遮挡情况复核。论文未说明代码、权重或数据发布链接；迁移时还需适配 RGB-D、位姿来源，并重新评估跟踪断裂与片段覆盖。

#### 总结
核心思想：二维先跟踪，三维再持久化
1. 稀疏分割创建掩码轨迹。
2. 稠密特征传播并在关键帧校正身份。
3. 将带置信度的掩码融合成三维片段。
4. 用几何与外观合并断轨及历史重访。
5. 以多视角 CLIP 特征检索目标。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.31005v1)
- [arXiv](https://arxiv.org/abs/2609.31005v1)

---

<a id='2609.31207v1'></a>
## [Enabling a Unified Cross-Domain Representation for Two-Finger Gripper Manipulation via Interaction-Centric Modeling](https://arxiv.org/abs/2609.31207v1)

**Authors:** Guanlin Li, Shifeng Bao, Yihan Zhao, Haitao Shen, Haoyang Li, Chen Zhao, Tong Yang, Jie Tang, Jing Zhang

**Published:** 2026-09-25

**Categories:** cs.RO, cs.CV

**Abstract:**

Achieving robust cross-embodiment generalization in imitation learning demands overcoming a critical representation flaw that inextricably entangles task semantics with hardware-specific visual geometry. We propose an interaction-centric framework that leverages the shared structure of two-finger grippers via a parameterized universal gripper abstraction, yielding a canonical gripper-frame representation. Given language and RGB-D observations, a VLM infers the subtask and grounds an interaction triplet (gripper, held, target), while SAM~2.1 tracks masks to reduce VLM queries. We design concise hybrid features that combine target/collision artificial potential fields for global guidance with segmented gripper-frame point clouds for local geometry, and use a Flow-Matching Transformer to predict smooth 7-DoF action chunks. Experiments in simulation and real-world tasks demonstrate that ours is the first imitation learning approach to simultaneously achieve competitive benchmark scores and extreme cross-embodiment/cross-viewpoint zero-shot sim-to-real transfer to completely distinct, heterogeneous robot platforms.

### 论文解读

#### 摘要翻译
模仿学习策略常把任务语义与机器人夹爪、视角等硬件几何纠缠，换平台后难以泛化。本文提出双指夹爪交互中心框架：用参数化通用夹爪建立规范坐标，视觉语言模型从语言和 RGB-D 中推断子任务及“夹爪—持有物—目标”三元组，SAM 2.1 跟踪掩码；再融合全局目标／碰撞势场与局部点云，由 Flow-Matching Transformer 输出平滑的 7-DoF 动作块。作者报告仿真和真实机器人结果，并展示跨具身、跨视角零样本迁移。

#### 方法动机分析
直接从图像学动作容易记住特定机器人外形，导致更换夹爪、相机或平台时失效。核心假设是不同双指夹爪可由尺寸参数和夹爪自身坐标统一描述，而操作任务可拆为当前夹爪、手中物体和下一目标的关系。于是策略先把观测变为共享的交互几何，再解码动作；它仍依赖单视角标定、可靠的 VLM/SAM 目标定位和两指夹爪，不能视为对任意具身的无条件不变。

#### 方法设计详解
输入为语言、固定相机 RGB-D、夹爪长宽高及末端 7-DoF 位姿。GLM-4.6V 每隔 5 步解析当前子任务与相关区域，SAM 2.1 分割目标并在其间跟踪。深度点云被变换到以夹爪为原点的坐标系：夹爪尺寸生成 8 个底座／指尖关键点；目标吸引势场与障碍排斥势场在关键点累积方向和对数幅值，形成 16 个四维特征，给出紧凑全局引导。与此同时，掩码提取持有物和目标点云，经最远点及距离加权采样各留 256 点，保留精细几何。冻结文本编码器配可训练适配器，PointNet 编码两组点云，Transformer 编码势场，MLP 编码位姿；各夹爪共享编码器。语义、几何、位姿和带噪动作 token 输入 8 层因果 Transformer，隐层 512、约 3000 万参数，以 Flow Matching 预测动作块。双臂各用一个夹爪区；单臂时固定未活动臂的区。策略以仿真示范训练，基准每项任务使用 50 条 Clean 示范；学习率、批量大小和训练轮数未说明。

#### 方法对比分析
与直接用原始点云或图像回归动作不同，本文的创新在于显式构造语义—几何中间层；相较对象中心表示，它突出交互关系而减少对完整物体 3D 标注的依赖。势场以少量关键点提供全局避碰／趋近方向，局部点云弥补其几何损失，因果掩码则维持动作块时间连贯。消融支持这些模块互补，但系统仍依靠语义分割和正确标定，并未消除感知误差。

#### 实验分析（精简版）
在 RoboTwin2.0 的 25 项筛选任务上，Clean 与随机化测试各 100 条；本文成功率为 66.3% 和 46.1%。随机化成绩高于 DP3-W 的 7.5% 与 BagelVLA 的 28.4%，但后两者 Clean 分别达 73.1% 和 85.5%。去掉势场后随机化成绩由 46.1% 降至 35.1%。仿真训练策略零样本迁移到真实机器人时，Shake Bottle 从仿真 97% 降至 DOS-W1 的 16/30、Jaka 近视角的 15/30 和大幅变视角的 9/30；相机 OOD 倒水任务各方法均为 0/10。结果显示抗随机化和一定跨具身能力，也暴露视角变化及精细操作的明显退化；作者未报告置信区间或显著性检验。

#### 实用指南
复现需有 RGB-D 标定、夹爪尺寸、示范数据和 VLM/SAM 语义分割链路；仿真评估为每项 50 条训练示范，动作模型约 30M 参数。论文估计非 VLM 部分约 10 ms；以本地 VLM 200–500 ms、每次语义更新覆盖 5 次跟踪和 100 个 20 Hz 动作为前提，可摊薄感知延迟。当前实现使用延迟不固定的闭源 GLM-4.6V API。跨平台须重新校准相机并匹配工作空间，尺寸或坐标不准可能造成对齐失败及逆运动学问题。论文未提供本文代码、模型或数据集的公开链接或明确开源声明。

#### 总结
核心思想：交互表征解耦夹爪具身差异。
1. VLM 定子任务与交互三元组，SAM 跟踪相关对象。
2. 夹爪尺寸统一坐标，构造全局势场和局部点云。
3. 因果 Flow-Matching Transformer 融合特征并生成平滑动作块。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.31207v1)
- [arXiv](https://arxiv.org/abs/2609.31207v1)

---

<a id='2609.31225v1'></a>
## [Imp-ACT: Adaptive Impedance Control and Action Chunking with Transformers to Learn Contact-Rich Manipulation from Demonstrations](https://arxiv.org/abs/2609.31225v1)

**Authors:** Luca Zanetti, Doganay Sirintuna, Idil Ozdamar, Pietro Balatti, Heng Zhang, Arash Ajoudani

**Published:** 2026-09-25

**Categories:** cs.RO

**Abstract:**

Contact-rich manipulation requires robots to balance accurate motion tracking with compliant interaction, yet most visual-action policies leave compliance fixed at the controller level. We present Imp-ACT, a methodologically grounded and practical approach to incorporating direction-dependent Cartesian stiffness modulation directly into demonstration collection, without manual stiffness selection or offline target reconstruction. During teleoperation, a self-tuning impedance controller adapts stiffness along the instantaneous direction of motion while maintaining compliance in orthogonal directions. The adapted stiffness is applied and recorded alongside visual observations and motion commands, capturing motion and compliance under the same dynamics. We implement this pipeline using Action Chunking with Transformer (ACT) to predict end-effector pose, gripper action, and motion-direction stiffness from visual, proprioceptive, and wrench observations. The performance of Imp-ACT is evaluated on wiping and plug insertion using both success rate and quantitative measures of contact behavior. Compared with fixed low- and high-stiffness baselines, Imp-ACT achieves comparable or higher success while maintaining low interaction forces. In wiping, it reduces contact-force vibration by approximately $29\times$ relative to the compliant baseline and $180\times$ relative to the stiff baseline. In plug insertion, it reduces forces orthogonal to the insertion direction by $43\%$ relative to the better fixed-stiffness baseline. These results highlight the benefit of maintaining sufficient stiffness along the direction needed for task execution while preserving compliance in other directions to limit contact forces and accommodate environmental constraints.

### 论文解读

#### 摘要翻译
接触丰富的操作需要准确跟踪与柔顺交互，但视觉动作策略通常把柔顺性固定在控制器里。Imp-ACT 在采集示教时自动调节并同步记录依赖运动方向的笛卡尔刚度，操作者只通过 VR 指定末端位姿和夹爪动作。基于 ACT 的策略再根据视觉、机器人状态和交互力，分块预测位姿、夹爪及方向刚度。在擦拭与插头插入任务中，它达到与固定刚度基线相当或更高的成功率；擦拭接触力振动约降低 29 倍和 180 倍，插入横向力较优固定基线降低 43%。

#### 方法动机分析
固定低刚度较安全却可能难以克服摩擦、精确跟踪；固定高刚度推进更快，但会增加冲击和错位接触负载。若示教后再改变顺应性，动力学与策略遇到的状态分布也随之改变。论文假设：沿运动方向需要较高刚度以推进和抵抗跟踪误差，而正交方向应保持柔顺以容纳表面或插座错位。关键是让策略从真实经历过的自适应示教中学到刚度，而非人工指定模式。

#### 方法设计详解
期望末端平移的相邻时刻差构成方向基 (U)，其主轴沿运动方向。平移刚度为 (U,diag(k_{st},k_{min},k_{min})U^T)：主轴刚度自适应，另外两轴维持最低值；阻尼按 (D=2\zeta\sqrt K) 随刚度配置。误差超过阈值时增加主轴刚度，跟踪误差较小且外力下降足够快时降低刚度，其余时间保持。实验设刚度范围 200–1000 N/m、正交刚度 200 N/m、旋转刚度 30 Nm/rad、(zeta=0.7)，误差门限 5 cm、力下降率门限 2 N/s。操作者用 VR 只控制目标位姿和夹爪；控制器施加顺应性并同步记录图像、TCP 位姿、夹爪、外力及实际刚度。策略输入腕部和固定相机的 480×640 图像、18 维本体/力状态，输出 11 维动作（位姿、夹爪、方向刚度），每块 50 步、约 1 秒；推理频率 50 Hz，使用 0.1 时间集成系数。示教使用 Franka Panda 与 Robotiq 夹爪，擦拭和插入各刚度条件分别采集 40 和 80 段，策略训练 300,000 步。

#### 方法对比分析
区别不只是把力觉加入策略，而是采集期就在线产生刚度标签，并让策略联合预测运动与刚度。论文指出 CRAFT、TacVLA 使用交互反馈但不预测顺应性；Comp-ACT、DIPCOM 需要手工选顺应模式；ACP 从统一低刚度示教离线重建顺应性；CompliantVLA-adaptor 则在部署时追加刚度选择。该思路适合推进轴与横向对准需求不同的接触任务；推广到 Diffusion Policy 或更广泛 VLA 尚属未来工作。

#### 实验分析（精简版）
每策略每任务评测 25 次。擦拭成功率：低刚度 96%、高刚度和 Imp-ACT 均 100%；8–40 Hz 竖直力振动功率分别为 17.7、109.6、0.6 N²，Imp-ACT 相比两基线约低 29 倍和 180 倍。插入成功率为 60%、72%、76%；横向力分别为 7.8、11.9、4.4 N，较优固定基线低约 43%。低越好指标仅统计成功轨迹。结果支持较低交互负载与有竞争力的成功率，但每项仅 25 次且只测两个任务；振动改善归因于方向刚度的解释仍是作者假设，并非单独消融证明。

#### 实用指南
复现需在示教控制器中实现误差增刚、力下降减刚，并记录实际施加的刚度；训练/部署要保持传感器、动作定义和刚度映射一致。论文未说明代码或数据是否开放，也未给出优化器、学习率和 batch size。迁移到新机器人或任务时需重标定运动方向、门限与阻尼，重新采集示教并训练；旋转刚度自适应和被动性过滤器仍待研究。

#### 总结
核心思想：把方向顺应性学进动作块。
1. VR 给出位姿和夹爪指令。
2. 控制器沿运动方向按误差、接触反馈调刚度并记录。
3. ACT 从图像、状态、力觉生成分块位姿、夹爪和刚度。
4. 执行预测动作，以学习到的刚度闭环交互。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.31225v1)
- [arXiv](https://arxiv.org/abs/2609.31225v1)

---

<a id='2609.31008v1'></a>
## [Quadruped Obstacle Avoidance and Footstep Planning with Distributed Low-cost Time-of-Flight Sensors](https://arxiv.org/abs/2609.31008v1)

**Authors:** Giammarco Caroleo, Timothée Mahamoodally, Matteo Manzardo, Jin Jin, Marco Pontin, Matias Mattamala, Renato Vidoni, Perla Maiolino, Maurice Fallon

**Published:** 2026-09-25

**Categories:** cs.RO

**Abstract:**

Quadruped robots typically rely on depth cameras and LiDAR sensors to map their local environment. However, these sensors have limited close-range coverage, are relatively expensive, and consume significant power. This study investigates whether distributed Time-of-Flight (ToF) sensors can serve as a low-cost alternative to depth cameras for near-field terrain mapping for locomotion and local navigation. We designed a distributed ToF sensing architecture for the ANYbotics ANYmal quadruped, assessed its environment reconstruction accuracy, and benchmarked it against depth cameras for terrain mapping and obstacle avoidance. Distributing these sensors around the robot can also avoid the blind spots of traditional sensors. Our results show that, despite their low resolution and higher measurement noise, distributed ToF sensors can support reliable perceptual locomotion with centimeter-level local mapping accuracy. The proposed sensing strategy provides sufficient geometric information for near-field obstacle avoidance and footstep planning, at substantially lower cost, energy consumption, and system complexity than depth cameras.

### 论文解读

#### 摘要翻译
四足机器人常用深度相机和激光雷达建图，但近处可能有盲区，设备成本与功耗也较高。本文为 ANYmal 设计分布式飞行时间（ToF）传感架构，并测试其几何重建、足步规划和障碍规避能力。结果显示，虽然测量较稀疏且噪声较大，ToF 仍能提供厘米级局部建图和近场任务所需几何信息，成本、能耗及系统复杂度更低。

#### 方法动机分析
单个小型 ToF 模块视野有限，作者的核心想法是把多个传感器分散安装并部分重叠，让机器人周围获得足以支撑短程运动的地形轮廓。目标不是替代 LiDAR 做长距离定位或全局建图，而是用较低代价补足腿脚附近感知。假设低分辨率地图仍能表达关键台阶和障碍；代价是小物体边界易漏、地图更粗糙。

#### 方法设计详解
ANYmal 前壳安装 8 个 ST VL53L5CX，每个输出 8×8 深度、视场角 65°、量程 0.02–4 m；阵列覆盖约水平 180°、垂直 90°。传感器经 I2C 串接至 RP2350 读出板，再由 ROS2 驱动生成点云，读出采样为 15 Hz，八路读出每周期约 1 ms；部分视场重叠有助于连续覆盖并缓解噪声或遮挡。重建评估另用环形 ToF 阵列与 Hesai LiDAR 做几何对照。CAD 模型提供外参，点云投影到机器人局部地图，由 elevation-mapping 框架以 40 Hz 更新；作者将 ToF 传感器模型参数设为 0.01、0.02、2、1，横向因子 0.014。状态估计器提供机器人位姿，ToF 仅建地形图，不负责定位。高程图交给已有感知式强化学习运动控制器规划足步，或交给反应式局部规划器绕障。

#### 方法对比分析
相较两台视场约 87°×58°的 RealSense D435i，分布式布置更针对机器人前方及脚边近场；单个 ToF 约 10 美元、0.2–0.4 W，论文给出的 D435i 对比约 400 美元、2–4 W。贡献主要在低成本传感阵列、硬件/软件集成和实机评估，并未提出新的规划算法。适合近场感知，不适用于要求全局一致地图或远距离探测的任务。

#### 实验分析（精简版）
与 LiDAR 点云对比，ToF→LiDAR 距离中位数为 1.9 cm、P95 为 17.4 cm，说明典型误差小但稀疏区域仍有长尾。楼梯足步试验中，ToF 和深度相机配置均完成 10/10 次；盲测机器人 3 次均失败。实机纸箱障碍测试每难度各 10 次，ToF 易、中、难场景分别失败 0、1、2 次，深度相机为 0、0、2 次；ToF 地图更稀疏，困难场景边界漏检会增加碰腿。结果支持特定近场任务可用，尚不能说明复杂户外或长期运行下与高分辨率传感器等效。

#### 实用指南
复现重点是 8 个传感器的安装朝向与 CAD 外参、15 Hz 采样及文中地图模型参数；地图位姿仍依赖机载状态估计和 LiDAR 里程计。论文未说明代码、数据公开链接，也未报告更细的软件滤波配置。迁移时需重做布置和标定，并在目标场景检查遮挡、薄小障碍漏检及地图更新延迟；远距离导航应融合其他传感器。

#### 总结
核心思想：分布式 ToF 补足脚边视野
1. 多点布置并重叠近场深度视野。
2. 将 ToF 点云按位姿投到高程图。
3. 把局部地图交给现有足步或避障控制器。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.31008v1)
- [arXiv](https://arxiv.org/abs/2609.31008v1)

---

<a id='2609.30889v1'></a>
## [PHASE: Compliance-Enabled Tactile Phase Retrieval for Few-Shot Insertion Learning](https://arxiv.org/abs/2609.30889v1)

**Authors:** Jeremy Siburian, Cristian C. Beltran-Hernandez, Tatsuya Matsushima, Yusuke Iwasawa, Masashi Hamaya, Mai Nishimura

**Published:** 2026-09-25

**Categories:** cs.RO

**Abstract:**

Contact-rich assembly tasks such as peg-in-hole insertion remain difficult to learn from limited demonstrations. While retrieval-augmented imitation learning, which augments target demonstrations with relevant prior data, offers a promising direction, its applicability to contact-rich manipulation remains largely unexplored. Contact-rich insertion unfolds over multiple phases from search to insert, and retrieving phase-specific experience from prior data in principled ways remains an open question. Our key insight is that a compliant wrist enables the robot to sustain contact throughout execution, producing rich tactile and force signals that naturally reveal the phase structure of insertion and inform what should be retrieved. Based on this insight, we present PHASE (PHase-Aware Segmentation and REtrieval), a framework for compliance-enabled tactile phase retrieval that integrates multimodal contact-aware representation learning, variable-length phase segmentation from tactile signals, and phase-consistent retrieval for policy learning. We evaluate PHASE on real-world peg-in-hole insertion across five peg geometries, comparing against retrieval strategies drawn from state-of-the-art methods under a shared policy architecture. PHASE improves the overall success rate by 13 percentage points over the strongest non-phase-aware baseline, and improves performance under unseen initial positions by 30 percentage points. These results demonstrate that aligning retrieval with interaction-defined contact phases substantially improves robustness in few-shot insertion learning.

### 论文解读

#### 摘要翻译
插入孔等接触密集装配任务仍难以从少量示范中学会。检索增强模仿学习可用相关先验数据扩充目标示范，但在接触操作中的应用仍不充分；如何从先验中有原则地检索搜索、插入等阶段经验也是开放问题。作者发现，柔顺腕能让机器人持续接触，触觉与力信号因而显露阶段结构。由此提出 PHASE：结合多模态接触表示、触觉驱动的变长阶段分割和阶段一致检索来学习策略。在五种形状的真实插入任务中，PHASE 比最强非阶段感知基线总体成功率高 13 个百分点；未见初始位置下高 30 个百分点。

#### 方法动机分析
插入的相位转变由接触主导，持续时间随几何形状与起始位姿变化。固定时间窗可能拆散连贯行为，或混合搜索和插入的不同接触机制；单看运动相似度也不能保证接触过程相似。方法的核心假设是：柔顺腕维持接触所产生的触觉/力信号，比运动学更能揭示可迁移的交互阶段。

#### 方法设计详解
整体 pipeline 的输入是多模态示范轨迹，输出是训练好的 ACT 插入策略；编码器、触觉阶段分割、同阶段检索和策略学习依次衔接。机器人以 3×3 三轴触觉阵列、腕部力/扭矩及本体状态记录轨迹。MAT³ 在先验示范上做掩码重建，得到 256 维逐帧嵌入；用 15 帧历史窗，掩码比例在 0–0.6 采样，训练 100 轮后冻结编码器。对触觉阵列各 taxel 的力按位置求力矩和，得到估计扭矩；在其模长峰值之后，以 15 帧窗口计算变异系数，首个 CV<0.1 的时刻作为搜索到插入的边界。扭矩窗口按 50 Hz 采样时长为 0.3 秒；若找不到稳定点，则取有界搜索区间末端。随后将搜索段和插入段分别用 FastDTW（半径 1、逐帧欧氏距离）与先验中同阶段片段比较，每个查询阶段取最近的 20 段。目标示范保留完整轨迹，检索片段独立作为训练轨迹，再共同训练 ACT 行为克隆策略；策略预测 50 步动作块，采用 32 维潜变量、AdamW、学习率 10⁻⁴、batch size 64，训练 100,000 步。

#### 方法对比分析
基线在相同策略框架中检索单个状态—动作对、30 帧固定窗或完整 episode。PHASE 的关键不是单纯使用 DTW，而是先用交互信号分段，再限定同阶段匹配变长片段，使训练覆盖接触转变而减少不相容阶段混合。它适合接触阶段清晰、示范间阶段时长变化大的插入任务；迁移到其他任务需重设阶段边界或学习新的接触表征。

#### 实验分析（精简版）
五种几何形状各测试 20 次，每方法共 100 次，以 60 秒内完全插入为成功。PHASE 总成功率为 77%（77/100），比 BC-Prior 的 64% 高 13 个百分点，比仅用目标示范的 50% 高 27 个百分点；单点、固定窗、整轨迹检索分别为 56%、55%、64%。在训练中未见的初始位置（水平偏移最多 ±2 cm）下，PHASE 为 47%（47/100），BC-Prior 为 17%（+30 个百分点），表明鲁棒性提升但大偏移恢复仍不充分。各检索策略训练帧数不同，PHASE 使用 15,548 帧，因而实验并未完全隔离结构与数据量影响。

#### 实用指南
先验集由方形与圆形任务的 122 条示范组成，每种目标形状另采 4 条示范；采集和控制频率均为 50 Hz。复现可从 15 帧扭矩稳定性检测、top-20 同阶段检索及上述 ACT 超参数入手。软腕和触觉阵列支撑当前力矩估计；换硬件需重新标定 taxel 力与位置。论文未说明 PHASE 自身代码、模型或数据是否开放。

#### 总结
核心思想：按接触阶段检索示范。
1. 柔顺腕采集持续接触信号。
2. MAT³ 编码触觉与本体轨迹。
3. 扭矩峰后稳定点切分搜索/插入。
4. FastDTW 检索同阶段变长片段。
5. 与目标示范一起训练 ACT。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.30889v1)
- [arXiv](https://arxiv.org/abs/2609.30889v1)

---

<a id='2609.30969v1'></a>
## [TACTIC: Understanding Tactile Encoders and Conditioning for Contact-rich Robot Manipulation Policies](https://arxiv.org/abs/2609.30969v1)

**Authors:** Seongjin Bien, Débora Oliveira Makowski, Carlo Kneissl, Reihaneh Mirjalili, Pankhuri Vanjani, Rudolf Lioutikov, Gitta Kutyniok, Florian Walter, Wolfram Burgard

**Published:** 2026-09-25

**Categories:** cs.RO

**Abstract:**

Tactile information is essential for contact-rich manipulation tasks in robotics. Vision-based tactile sensors make it particularly easy to design end-to-end manipulation policies with tactile sensing, as they enable the use of existing encoders from computer vision. However, this has led to a huge variety of architectures, training datasets, and evaluation protocols, making it difficult to determine which design choices best encode touch. In this work, we address this gap and present a comprehensive study of tactile encoders and fusion strategies across various contact-rich manipulation tasks in real-world experiments. To enable a controlled comparison, we train and evaluate all models under the same pipeline and experimental setup, comprising more than 2000 real-world rollouts. Our results go beyond other studies that only compare simulation performance, which does not necessarily translate to real-world settings, where large-scale evaluations are needed to obtain reliable statistics. Our key finding is that there is no universally optimal representation or fusion strategy for encoding visual-tactile. Instead, the best encoder backbone and fusion scheme depend strongly on the task.

### 论文解读

#### 摘要翻译
触觉对接触密集型机器人操作至关重要。视觉式触觉传感器便于沿用计算机视觉编码器，但既有工作在架构、数据与评测协议上的差异，使设计选择难以比较。TACTIC在统一流程中比较真实机器人任务里的触觉编码器和融合策略，完成超过两千次执行。结果表明，视觉—触觉表示没有通用最优解，最佳编码器与融合方式取决于任务。

#### 方法动机分析
仿真中的接触未必能迁移真实环境，小规模实机测试也难支撑可靠结论，构成评估不足；只比较触觉策略和视觉基线，又无法辨别编码与融合的作用。作者因此控制机器人、策略和评测流程，考察不同接触需求是否偏好不同触觉线索。

#### 方法设计详解
系统以双Franka机械臂遥操作采集示范，从臂夹爪装左右DIGIT触觉传感器，并配腕部和侧视相机。四项任务分别是擦板、拧螺母、插USB-C和擦曲面花瓶，约每项100条示范、采样率30 Hz。输入包括两路触觉图像、两路视觉图像及关节/夹爪状态；通常从触觉图像扣除每次运行前50张空白帧的背景，T³则使用原始图像。每个任务单独训练ACT，预测未来120步关节位置增量动作块，并对重叠预测作时间加权集成。擦拭与拧螺母系数为0.01，USB-C为0.1。
五种触觉编码器包括ImageNet ResNet-18、示范数据自监督预训的SARL、VQ-GAN量化的UniT、跨时刻t与t−4输入的Sparsh-DINO，以及保持预训练权重的T³。五种融合包括特征拼接Concat、由夹爪开合状态调制通道的FiLM、以触觉查询视觉键值并门控残差的GCA，以及视觉—触觉对比学习CLIP-R/CLIP-T。CLIP-R每批256样本、跨任务随机采样；CLIP-T每批32样本、同轨迹至少间隔30帧采样。专用预训练触觉骨干在ACT训练时冻结。

#### 方法对比分析
TACTIC的主要贡献是统一实机协议下的5种编码器、5种融合与4种任务比较，而非单一新策略。Concat简单但有竞争力；FiLM注入夹爪接触先验，GCA按视觉信息调制触觉，CLIP重新对齐两种模态。比较显示，法向力控制、滑移检测和精细插入需求不同，复杂融合并不必然优于拼接。

#### 实验分析（精简版）
主要实验每个策略条件进行20次实机执行，总计报告2,180次（含分布外评估）。Sparsh-CLIP的平均终止成功率接近50%，但在擦板任务明显失利；USB-C插入中ResNet18-Concat达到80%，视觉单模态基线为55%，更复杂融合反而降低该编码器表现。拧螺母时Sparsh-CLIP-T与T³-CLIP-T均达90%；擦花瓶时UniT-CLIP-T和UniT-GCA均为55%。分布外每种选定条件测试10次：有干扰物时Sparsh-CLIP-T拧螺母仍为90%，擦花瓶为50%；变化灯光下拧螺母也维持90%。优势是揭示任务专长，局限是分布外仅覆盖两种人工视觉变化，且CLIP效果可能受相机配置影响。

#### 实用指南
论文提供TACTIC项目页，并将任务材料、训练流程和评估配置列为开放贡献；示范数据及所有模型权重的发布范围未明确。复现需保留DIGIT背景处理、夹爪状态采样、ACT动作块设定，并按每项任务重新训练和比较融合策略；论文未说明优化器、学习率和训练轮数。

#### 总结
核心思想：融合策略要按任务选择
1. 遥操作采集多类接触任务示范。
2. 交叉组合触觉编码器与融合模块。
3. 用单任务ACT预测动作块并实机评估。
4. 结合分布外结果选择任务适配方案。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.30969v1)
- [arXiv](https://arxiv.org/abs/2609.30969v1)

---

<a id='2609.31112v1'></a>
## [DualManip: Agentic Dynamic Manipulation via Dual-Path Semantic Reasoning and Geometric Adaptation](https://arxiv.org/abs/2609.31112v1)

**Authors:** Chengxi Li, Yan Di, Yingyue Li, Ruida Zhang, Mingyang Li, Xiangyang Ji

**Published:** 2026-09-25

**Categories:** cs.RO

**Abstract:**

Vision-language models (VLMs) enable open-vocabulary reasoning for robot manipulation, but their high inference latency limits responsiveness in dynamic scenes. Many scene changes, however, alter object geometry without invalidating task intent. We present DualManip, a dual-path framework that decouples infrequent semantic reasoning from responsive geometric adaptation. The semantic path decomposes the task and grounds task-relevant interactions, followed by a constraint-solving module for pose optimization. During execution, the geometric path continuously updates template-to-observation correspondences from live RGB-D observations via a shape-adaptive network. These correspondences transfer task-relevant grasp contacts across observations, enabling online grasp reconstruction under object motion and non-rigid deformation. The Information Interaction Module bridges the two paths by initializing task-relevant grasps from semantic grounding, validating geometric updates, and triggering semantic replanning upon update failures. Real-world evaluation spans six manipulation tasks covering non-rigid deformation, articulated reconfiguration, rigid motion, and high-precision assembly across three settings: static, single-change, and continuous dynamic. DualManip demonstrates superior manipulation robustness, particularly under continuous scene changes, while achieving geometric adaptation approximately 46$\times$ faster than agentic verification and semantic replanning. Our project page: https://lichengxi1.github.io/Dualmanip.

### 论文解读

#### 摘要翻译
DualManip 面向动态机器人操作，将低频的视觉语言语义推理与高响应几何适应分开。语义路径分解任务、定位交互区域并求解位姿；几何路径依据实时 RGB-D 和形状自适应对应网络，把任务相关抓取接触点从模板迁移到运动或变形后的物体。更新不可靠时，信息交互模块才触发语义重规划。论文在六项真实操作任务和静态、单次变化、连续动态三种设置下验证，覆盖非刚性、关节、刚体及精密装配场景。

#### 方法动机分析
动态场景的关键痛点是“几何变了，但任务意图没变”。每次变化都调用 VLM 会产生高延迟，完全不更新又会执行过时抓取；刚体位姿和稀疏点跟踪也难处理连续变形。DualManip 假设语义意图仍有效时只需更新几何，只有对应失败、不可达或碰撞风险才重新理解任务。

#### 方法设计详解
输入为指令和 RGB-D。VLM 将任务拆成阶段，SAM 3 得到物体与功能区域掩码，深度恢复点云，Sofar 推断任务条件方向；预定义距离、垂直、共线等关系形成约束，并优化末端位姿。每个物体先由多视角重建模板，模板和当前点云都下采样到 1,024 点。对应网络经特征相似度与 Sinkhorn 得到软匹配，同时预测模板形变和刚体变换；配对、Chamfer、三维、投影及 ARAP 正则损失共同训练。AnyGrasp 在语义掩码内初始化抓取，两个接触点通过共享模板对应迁移，重建中心、闭合方向和姿态，再检查可达性与碰撞。每个物体每个相机视角收集 100 个 RGB-D 观测用于训练。

#### 方法对比分析
ReKep 依赖稀疏关键点，OmniManip 依赖物体级刚体 6D 位姿，对强变形和关节变化受限；CLEA 通过视觉验证和高层重规划适应变化，但难以实时跟随。DualManip 的区别是把两个任务相关接触点放入共享模板空间连续迁移，以几何路径保持意图，失败时才调用语义路径。因此它适合有模板、RGB-D 质量可控且物体状态持续变化的操作，不是完全零样本的未知类别方案。

#### 实验分析（精简版）
平台使用 KUKA iiwa、Robotiq 夹爪、两台 ZED 2i 和 RTX 3090，每种组合 15 次。静态一般任务 DualManip 为 77.8%，与最佳基线持平；静态装配为 51.1%，高于 OmniManip 的 40.0%。连续变化下平均成功率为 53.3%，而 ReKep 为 28.9%、OmniManip 为 17.8%，CLEA 为 0.0%。单次更新延迟为 138.7 ms，CLEA 使用 GPT 为 6.376 s，约慢 46 倍。局限是连续变化仍有失败，且模板与物体特定训练限制泛化。

#### 实用指南
复现需为每个物体构建模板，采集各相机视角 100 个 RGB-D 观测，统一使用 1,024 点，并实现 Sinkhorn 对应、MARCO 伪标注、多项几何/投影损失和碰撞检查。论文给出损失权重，但未说明骨干、优化器、学习率、epoch 或 batch size。项目页已给出链接，不过正文未明确代码、模型和数据是否开放。迁移到新物体需替换模板并训练对应网络，迁移到新机器人还需重做运动学、抓取和碰撞适配。

#### 总结
核心思想：语义意图与几何适应解耦
1. VLM 分解任务并定位交互区域。
2. 约束求解和 AnyGrasp 初始化语义抓取。
3. 模板对应网络联合适应形变与运动。
4. 迁移接触点并验证可达性、碰撞。
5. 失败时才触发语义重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.31112v1)
- [arXiv](https://arxiv.org/abs/2609.31112v1)

---

<a id='2609.30951v1'></a>
## [Bundled Contact Gradients: Stabilizing Differentiable Simulation for Deployable Dynamic Tasks](https://arxiv.org/abs/2609.30951v1)

**Authors:** Dyuman Aditya, Jin Cheng, Clemens Schwarke, Quan Nguyen, Gaurav Sukhatme, Stelian Coros, Gabriele Fadini

**Published:** 2026-09-25

**Categories:** cs.RO

**Abstract:**

Differentiable simulation provides analytic gradients of robot dynamics, enabling fast and sample-efficient first-order policy optimization. However, obtaining smooth and informative gradients through rigid-body contact typically requires softened contact models, often at the expense of physical fidelity and thereby limiting learned policies largely to simulation. This trade-off becomes particularly consequential for dynamic humanoid motions, where accurate contact dynamics are critical for transferring policies to the real world. Increasing contact stiffness in rigid-body simulation improves the fidelity of interactions, but also makes the dynamics increasingly sensitive to small state perturbations, producing high-variance gradients that can destabilize first-order policy learning. To address this, we propose \emph{Bundled Contact Gradients (BCG)}, a contact-local randomized smoothing framework for differentiable policy learning. When stiff contact is detected, our method evaluates a local bundle of randomized perturbation rollouts around the stiff contact configuration and aggregates their gradient signal thereby reducing gradient variance. We demonstrate the effectiveness of our method by successfully training and transferring dynamic motions zero-shot onto a real-world Unitree G1 humanoid platform. Videos and supplementary information can be found at https://bundledcontactgradients.github.io/

### 论文解读

#### 摘要翻译
可微仿真能提供机器人动力学解析梯度，支持高样本效率的一阶策略优化，但软化接触会牺牲物理真实性，难以迁移到真实机器人。提高接触刚度虽更真实，却使梯度对微小状态扰动敏感。作者提出捆绑接触梯度（BCG），在高刚度接触附近采样扰动轨迹并聚合梯度，降低方差。方法在Unitree G1上实现动态动作的零样本真机迁移。

#### 方法动机分析
PPO等零阶方法能处理非光滑接触，却需要大量环境交互；SHAC等一阶方法更高效，但高刚度接触会带来冲突、尖锐的梯度。全局平滑成本高，接触处截断又丢失关键动力学信息。BCG假设接触响应仍是连续且局部可微的高刚度模型，只在真正脆弱的接触局部平滑。

#### 方法设计详解
输入是高刚度接触状态，输出是聚合后的状态、奖励和梯度。仿真器用有限刚度的解析平滑接触模型计算状态转移：检测到法向接触力超过400 N时，在接触末端的笛卡尔位置、速度上采样高斯扰动，再用运动学雅可比的阻尼伪逆映射到关节状态。B=10个分支共享策略动作，各自推进H=2个控制步；位置和速度取平均，聚合状态继续驱动策略、奖励和critic。分支敏感度也按聚合器反传，接入SHAC的短时域actor–critic；ADD以参考与模拟运动特征残差生成可微模仿奖励。训练设置为N=32控制步的rollout，GPU并行分支，扰动尺度为1 cm和2 cm/s。

#### 方法对比分析
BCG不是把接触整体软化，也不是在接触事件处截断梯度，而是保留高刚度前向动力学，并对邻近接触实现的导数求局部平均。相比全局随机平滑，它只在接触触发，GPU并行限制了额外开销；相比PPO，它利用动力学解析梯度，适合接触密集且希望样本高效的动态模仿。持续接触或需要严格不连续硬接触的任务仍可能受限。

#### 实验分析（精简版）
任务为LAFAN1的跑步、跳跃、格斗和舞蹈，每段15秒，在Warp训练、MuJoCo评估。刚度为300时四种动作迁移均0/5跌倒，足部穿透差2.9–3.4 mm；刚度为50时均5/5跌倒。迁移跟踪误差上，BCG相对PPO在Run为34.1±0.4对42.8±0.7 cm、Fight为13.5±0.3对34.3±1.4 cm，四项中三项更低。BCG达到相当奖励所需样本比PPO少一个数量级以上，但反向图带来额外计算/显存开销；真机结果为未微调的定性成功，未给出成功率。

#### 实用指南
复现需实现解析平滑接触、法向力触发、笛卡尔扰动及雅可比伪逆、共享动作的分支rollout和状态/梯度聚合，并结合SHAC与ADD。论文给出项目视频/补充资料网址，但未明确提供BCG代码或数据下载链接。迁移到其他机器人需替换接触链、动力学和参考特征，并重新调节刚度、阈值、分支数与扰动尺度；持续接触任务要预留更高计算和内存预算。

#### 总结
核心思想：接触邻域平均稳定高刚度梯度。
1. 用有限高刚度接触保持真实交互。
2. 检测接触并采样邻近位置、速度。
3. 多分支共享动作推进后聚合状态与敏感度。
4. 将稳定梯度用于SHAC+ADD并验证仿真、真机迁移。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.30951v1)
- [arXiv](https://arxiv.org/abs/2609.30951v1)

---

<a id='2609.31357v1'></a>
## [Transformer-based Monte Carlo Localization in Construction Meshes](https://arxiv.org/abs/2609.31357v1)

**Authors:** Linus Kramer, William Talbot, Olga Vysotska, Marco Hutter

**Published:** 2026-09-25

**Categories:** cs.RO

**Abstract:**

To be able to perform inspection or digitization tasks, mobile robots on construction sites must be able to localize themselves reliably with respect to a global reference frame that is shared with a building map. Similar room layouts and low-texture surfaces pose a challenge for existing LiDAR- and vision-based localization methods. We approach this problem with a LiDAR-based global relocalization system that estimates the robot's pose relative to a building mesh and combines a PointNet++ encoder with a place recognition decoder, whose outputs serve as a learned observation model within a Monte Carlo Localization (MCL) framework. The pipeline is trained exclusively on synthetic LiDAR scans obtained by simulating the robot's sensors inside the building mesh. Our approach is robust in ambiguous environments due to an uncertainty-aware decoder that scales positional likelihoods and a resampling strategy that injects model hypotheses into the particle set, enabling recovery from potential particle depletion. Evaluations on real-world datasets show that our method outperforms both diffusion-based and ScanContext++ baselines while maintaining fast inference (18 ms per call), demonstrating the practicality of synthetic-data training for mesh-referenced global localization in construction robotics.

### 论文解读

#### 摘要翻译
本文面向施工场地的全局定位，提出将 PointNet++、Transformer 多假设位置预测与蒙特卡洛定位（MCL）结合的方法。模型从一次 LiDAR 扫描预测多个位置及不确定性，再由粒子滤波持续跟踪。系统还预测观测置信度，并在重采样时注入新的位置假设，以缓解错误模式坍缩。模型只用建筑网格生成的合成扫描训练，却在四个真实数据集上有效，单次推理耗时 18 ms。

#### 方法动机分析
施工环境没有稳定 GPS，且存在重复房间、走廊、楼梯和裸露混凝土；网格与现场变化也会造成仿真到现实的差异。传统 LiDAR-SLAM依赖初始位姿，单点回归难以表达多种可能位置，普通 MCL 又可能过早集中到错误模式。论文假设建筑网格和 LiDAR 参数足以生成可迁移的训练数据，并用多假设分布保留歧义。

#### 方法设计详解
先从网格提取可站立位置：水平网格间距 0.1 m，并用 2–10 m 的上方空间筛选地面；模拟 LiDAR、加入方向和距离噪声，保留 0.5–20 m 点并下采样为 4096 点。PointNet++ 输出 256 维特征，Transformer 解码器使用 5 个查询、4 层、4 个注意力头和 10% dropout，输出 5 个三维高斯分量的均值、权重和标准差。训练采用位置负对数似然；置信度 β 将高斯混合与有效网格上的均匀密度融合。推理时用 1000 个六自由度粒子，权重为均匀项与预测密度的加权和；systematic resampling 后把 5 个粒子替换为预测高斯均值，再用半径 1 m、最小簇大小 100 的 DBSCAN 取最大簇质心。

#### 方法对比分析
ScanContext++ 将扫描压成极坐标高度描述子并检索数据库，MCL 版本只能把前 5 个匹配转换为固定方差、均匀权重的混合分布；Diffusion 通过多次随机去噪取得多个假设。本文用一次前向传播直接输出 5 个连续高斯假设，并预测方差与置信度，再把它们作为 MCL 的学习观测模型。核心创新是重采样后将预测均值注入粒子集，使错误模式可被新假设重新占据；这也是与标准 MCL 的关键区别。PointLoc 一类直接回归方法则只输出单一位姿。连续 MCL 的 1 m 召回率在 Hilti 为 91%，高于 Diffusion 的 43% 和 ScanContext++ 的 29%；Aesch 为 42%，高于 14% 和 0%。Transformer 单次推理 0.018 s，Diffusion 为 0.11 s。代价是需要建筑网格，现场变化过大时可能失配。

#### 实验分析（精简版）
实验使用 Hilti、Apartment、Stairwell、Aesch 四个真实数据集，连续 MCL 的 1 m 召回率分别为 0.91、0.86、0.73、0.42；Diffusion 分别为 0.51、0.69、0.51、0.14。去掉置信度后，本文模型在四集的 1 m 召回率为 0.49、0.81、0.61、0.33，加入置信度分别提升 42、5、12、9 个百分点。训练使用 Adam、学习率 1e−4、batch size 16；Aesch 仍有 4.44 m 平均误差，说明大规模重复结构是明显难点。

#### 实用指南
复现需要建筑网格、LiDAR 扫描模式和量程，按文中的可站立点提取、噪声模拟及 4096 点下采样生成训练数据；训练轮数随数据集为 400–1000。迁移到新场地需重新生成合成扫描并训练，校准传感器噪声、运动模型和网格可行区域。论文未提供明确的代码或数据开源声明；其评测轨迹由离线 Open3D SLAM 估计，不能直接视为在线定位组件。

#### 总结
核心思想：用多假设网络重启粒子滤波。
1. 从建筑网格模拟带噪 LiDAR 扫描。
2. PointNet++ 编码，Transformer 输出 5 个位置高斯及置信度。
3. 将预测分布与均匀先验融合，更新 1000 个 MCL 粒子。
4. 重采样后注入高斯均值，聚类得到最终位置。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.31357v1)
- [arXiv](https://arxiv.org/abs/2609.31357v1)

---

