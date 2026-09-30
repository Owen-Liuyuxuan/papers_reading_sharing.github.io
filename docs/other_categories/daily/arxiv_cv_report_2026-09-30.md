time: 20260930

# Arxiv Computer Vision Papers - 2026-09-30

## Table of Contents

1. [Degeneracy-Orthogonal Geometric Constraints for LiDAR SLAM](#2609.36753v1)
2. [A robust single-sensing-element tactile sensor for concurrent pressure and tackiness detection with real-time signal decoupling capability](#2609.36558v1)
3. [VidAct: Learning Manipulation from In-the-Wild Videos with Object-Centric 3D Awareness](#2609.36870v1)
4. [RoboChrono: A Real Robot Benchmark for Streaming Task Understanding](#2609.36605v1)
5. [OTRetarget: Joint Robot and Object Motion Retargeting via Optimal Transport](#2609.36602v1)
6. [HACo: Learning Haptic Active Compliance for Force-Aware Dexterous Manipulation](#2609.36596v1)
7. [Cooperative Multi-Agent Vision-Language-Action Models via Reinforced Fine Tuning](#2609.36588v1)
8. [Trajectory-Level Mode Guidance for Controllable Diffusion-Based Multi-Robot Motion Planning](#2609.36530v1)
9. [GlassFormer: Learning Real-time Glass Segmentation using Radar-Depth Fusion](#2609.36844v1)
10. [RoXDrive: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving via Action-Faithful Rollouts](#2609.36851v1)

---

## Papers

<a id='2609.36753v1'></a>
## [Degeneracy-Orthogonal Geometric Constraints for LiDAR SLAM](https://arxiv.org/abs/2609.36753v1)

**Authors:** Minseo Kim, Yina Kim, Jinhwa Hwang, Alex Junho Lee

**Published:** 2026-09-29

**Categories:** cs.RO

**Abstract:**

Autonomous robot navigation relies on simultaneous localization and mapping (SLAM) to estimate motion and maintain an accurate pose within an environment. However, in axially uniform corridors such as long tunnels and pipelines, LiDAR odometry is fundamentally limited by unconstrained drift along the feature-weak travel direction. This structural degeneracy cannot be resolved by local scan matching alone. To address this challenge, we propose the Degeneracy-orthogonal Contour Offset Descriptor (DeCOD), a structure-aligned geometric descriptor for cross-sectional landmarks. Cross-sectional boundaries, such as pipe joints and structural rings, provide metric constraints along this degenerate axis, but distinguishing individual landmarks requires capturing subtle surface variations across nearly identical profiles. The descriptor parameterizes signed normal deviation from estimated boundary contours, and matching explicitly resolves heading ambiguity and decouples first-order contour errors by distortion estimation. Matched landmarks yield geometric factors that enforce agreement in cross-section position and corridor axis alignment during pose-graph optimization, correcting longitudinal drift while leaving rotation about the common axis unconstrained. On a public benchmark and in field experiments, DeCOD achieves robust landmark retrieval over standard 3D descriptors and successfully stabilizes trajectories across different odometry frontends, reliably constraining longitudinal drift under geometric degeneracy.

### 论文解读

#### 摘要翻译
自主机器人导航依赖同步定位与建图（SLAM）估计运动并维持准确位姿。然而，在长隧道和管道等轴向均匀走廊中，LiDAR 里程计受限于弱特征行进方向上的无约束漂移，局部扫描匹配无法消除这种结构退化。本文提出退化正交轮廓偏移描述子 DeCOD，用于描述横截面地标。管接头、结构环等边界可提供退化轴向的度量约束，但区分近乎相同的地标需要捕捉细微表面变化。描述子参数化相对估计边界轮廓的有符号法向偏差，匹配时消解朝向歧义并补偿一阶轮廓误差。匹配地标形成几何因子，在位姿图优化中约束横截面位置和走廊轴线一致性，修正纵向漂移，同时不约束绕公共轴的旋转。公共基准和实地实验表明，相比标准三维描述子，DeCOD 的地标检索更稳健，并能跨里程计前端稳定轨迹，在几何退化下约束纵向漂移。

#### 方法动机分析
均匀墙面无法观测轴向位移；抑制不可靠更新只能防发散，不能恢复丢失的信息。核心假设是：横向接缝可重复检测，其细微表面差异足以辨认身份。方法依赖重访和可靠地标，不解决首次穿越的绝对定位。

#### 方法设计详解
输入短时累积点云与里程计。PCA 估计走廊轴，拟合正交截面轮廓，以轴向 DoG 和非极大值抑制检测边界。将邻域展开为轴向坐标 z 与归一化周长坐标 p，网格记录有符号法向偏移。

匹配前最小二乘消除周向偏置、轴向偏置及随 z 变化的倾斜项，保留细微形变；只比较共同可见区域，同时检验正反朝向。局部补丁互为最近邻匹配后做几何验证，以 S=2K/(N_i+N_j) 排序，仅接收唯一最高分且内点数达标的候选。输出位置与轴线对齐因子，联合里程计优化；不强制轴向旋转，避免错误姿态约束。

#### 方法对比分析
不同于全局场景描述和三角关键点方法，DeCOD 专门辨认提供弱方向约束的横截面地标。展开借鉴虹膜表示，创新在连续偏移、误差补偿和退化适配因子的组合；PCA、最近邻及鲁棒优化是标准组件。

#### 实验分析（精简版）
GEODE 与七条实地管道序列采用去程建库、回程查询；各方法使用相同局部点云，候选间隔至少 30 秒且不共享扫描。

1. Shield 上，2.4 米容差、至少 15 内点时，验证使精确率由 11.7% 升至 95.6%，但召回率仅 31.3%，体现保守取舍。
2. 使用至少 9 内点校正轨迹，S8-Beta 的 BIEVR-LIO 位置 RMSE 从 61.597 米降至 0.768 米；S9 仍残留 24.819 米误差，说明地标不足限制纠偏。

#### 实用指南
无需训练。推理时每隔 8 秒累积 20 帧，体素为 5.5 厘米；描述子为 16×256，轴向覆盖 ±2.4 米，补丁为 9×33。需重力参考、运动补偿和几何验证。RTX 4070 Ti SUPER 上查询 20 个候选耗时 69.7 毫秒，不代表完整 SLAM 延迟。迁移到窄管需调整空间尺度；代码开放状态及部分验证实现细节论文未说明，GEODE 为公共数据集。

#### 总结
核心思想：用横截面地标补足轴向约束
1. 沿退化轴检测边界并展开法向偏移。
2. 消除轮廓误差，双朝向匹配可见补丁。
3. 以几何共识筛选唯一可信地标。
4. 注入位置与轴线因子，保留轴向旋转自由度。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36753v1)
- [arXiv](https://arxiv.org/abs/2609.36753v1)

---

<a id='2609.36558v1'></a>
## [A robust single-sensing-element tactile sensor for concurrent pressure and tackiness detection with real-time signal decoupling capability](https://arxiv.org/abs/2609.36558v1)

**Authors:** Ying Yang, Mingwei Gu, Jia-Sen Xie, Xingyu Ma, Yan-Na Lu, Lin Zheng, Jinhui Gu, Junshuai Chen, Yunjie Lu, Denys Makarov, Jin Ge

**Published:** 2026-09-29

**Categories:** cs.RO, cond-mat.mtrl-sci

**Abstract:**

Integrating tackiness sensation into the artificial skin of humanoid robots significantly enhances their cognitive and operational capabilities. However existing tactile sensors face challenges in decoupling of the multimodal signal and stability. Here we present a surface-soft tactile sensor that incorporates a Hall effect sensor and a soft magnetic composite within a robust elastic framework. The sensor surface indents under pressure and bulges prominently when retracted from sticky surfaces dynamically altering the Hall sensor-magnet distance. This generates whole-process-traceable and baseline-separated signals enabling real-time differentiation between pressure and pull-off force. This single-sensing-element design facilitates bimodal sensing at the same contact spot while eliminate stress cross-talk enhancing both accuracy and sensitivity. The fusion of a robust framework and magneto-mechanical sensing mechanism equips the sensor with exceptional reliability and excellent signal baseline stability. This tactile sensor holds substantial potential for advancing robotic capabilities in evaluating adhesive properties monitoring rubber aging precisely handling lightweight objects and cognizing natural objects surface characteristics.

### 论文解读

#### 摘要翻译
将黏性感知集成人形机器人的人工皮肤，可显著增强其认知与操作能力。然而，现有触觉传感器面临多模态信号解耦与稳定性挑战。本文提出表面柔软的触觉传感器，在坚固弹性框架内集成霍尔传感器与磁体。受压时表面内凹，从黏性表面回撤时明显外凸，动态改变霍尔传感器与磁体的距离，产生全程可追踪、以基线分隔的信号，实现压力与拉脱力的实时区分。单传感元件设计支持同一接触点的双模态感知，并消除应力串扰，提高准确性和灵敏度。坚固框架与磁机械感知机制相结合，赋予传感器出色的可靠性和信号基线稳定性。该传感器有望用于评估黏附性质、监测橡胶老化、精确操作轻质物体及认知自然物体的表面特征。

#### 方法动机分析
黏附受预压力、接触时长与回撤速度共同影响，必须连续测量同一点的双向力。叠层结构易被拉脱，横向集成无法同点测量，部分共享膜方案又存在信号重叠。核心假设是：以弹性膜的双向位移编码力的方向，可直接实现物理解耦。

#### 方法设计详解
输入法向力，经PDMS顶柱传至共享膜，带动软磁体移动，改变霍尔输出。受压阶段顶柱向下推动膜，磁体靠近霍尔元件；从黏性表面回撤时膜反向外凸，磁体远离元件，因而同一接触点即可连续记录接触和拉脱全过程。定义ΔV/V₀=(V−V₀)/V₀：受压时磁体靠近、信号为正；受拉时远离、信号为负；卸载回零。基线两侧的符号直接编码力向，无需分类网络即可辨别方向，但定量测力仍需标定。推理时以传感器电压信号实时判向。
结构优化选择支撑环高1.4毫米、磁体厚0.7毫米；腔内海绵增强抗压，尽量保留拉力灵敏度。环己烷处理去除未反应单体，降低迟滞。

#### 方法对比分析
贡献不是霍尔元件本身，而是共享膜、稳固框架与磁场距离编码的协同：同点连续测力，同时利用基线两侧分离信号。海绵扩量程属于工程增强。适用于黏性检查与轻物放置，不等同于通用多轴触觉感知。

#### 实验分析（精简版）
实验包含结构参数扫描、重复压拉、耐久及机器人操作，无机器学习数据集。
1. 海绵使同型结构的饱和压力从1.3增至150千帕，约115倍，代价是压力灵敏度下降；拉力量程为0–33千帕。
2. 约63千帕下经历50000次循环，压力响应与基线保持稳定；快速回撤时可排序七种PDMS黏性。
慢速回撤下，最黏样品的峰值拉力反而较低，说明峰值不能普遍替代黏附能；强黏附仍可能损坏上部结构。

#### 实用指南
代码与数据开源状态论文未说明；无需训练模型。复现需PDMS/NdFeB软磁体、霍尔元件及力学标定设备；推理时通常供电5伏。磁化场约2.9特斯拉；响应时间测试用6伏。评估须控制接触面积、压力、驻留时间及回撤速度。迁移到其他机器人需适配安装、重新标定；微型化受霍尔封装限制。

#### 总结
核心思想：双向形变驱动基线解耦
1. 共享弹性膜将压力与拉力变为相反位移。
2. 磁体随膜移动，将位移转为磁场变化。
3. 霍尔信号在基线两侧区分力向。
4. 控制接触与回撤条件，以拉脱响应判断黏性。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36558v1)
- [arXiv](https://arxiv.org/abs/2609.36558v1)

---

<a id='2609.36870v1'></a>
## [VidAct: Learning Manipulation from In-the-Wild Videos with Object-Centric 3D Awareness](https://arxiv.org/abs/2609.36870v1)

**Authors:** Hang Li, Mingxin Zhang, Zihan Wu, Yang Tian, Dong Chen, Fengyi Shen, Yuan Meng, Xiangtong Yao, Heng Zhang, Ziyuan Liu, Zhenshan Bing, Alois Knoll

**Published:** 2026-09-29

**Categories:** cs.RO

**Abstract:**

Video demonstrations offer a scalable alternative to costly robot data for learning manipulation, yet existing reconstruction-based approaches often rely on constrained camera viewpoints or human-to-robot retargeting, while the reconstructed trajectories are difficult to adapt to new objects configurations without distorting the trajectory shape. Another key limitation is that the resulting policies often lack precise object-level 3D geometry awareness, limiting object grounding and object shape awareness critical for precise manipulation. To bridge these gaps, we propose VidAct, an efficient video-to-robot framework that learns object-centric, 3D-aware manipulation policies from a single monocular video per task and enables zero-shot real-world deployment. VidAct consists of three key components. First, VidAct reconstructs object meshes and motion from arbitrary demo videos and canonicalizes the motion in the static object frame, avoiding embodiment-specific retargeting and accommodating diverse camera viewpoints. Second, VidAct employ residual trajectory transfer for adapting the reconstructed motion to novel object configurations while preserving its motion shape. Finally, as the key policy-learning component, VidAct predicts simulation-provided privileged complete-object point clouds at each frame as an auxiliary task while retaining RGB-only deployment, providing dense object-centric supervision over both object pose and 3D geometry. Experiments on human, robot, generated, and internet videos demonstrate broad video applicability and zero-shot deployment. Per-frame complete-object 3D supervision improves policy generalization and sim-to-real success, while residual trajectory transfer enables reliable trajectory adaptation with better shape preservation.

### 论文解读

#### 摘要翻译
VidAct 将每个任务的一段单目视频转化为以物体为中心、具备三维感知的操作策略，并支持零样本真实部署。现有视频重建常受视角限制、依赖人机动作重定向，轨迹适配新物体配置时还容易变形；策略也缺少精确的物体级三维几何。VidAct 重建物体网格与运动，在静态物体坐标系中规范化轨迹；用残差轨迹迁移适配新配置；再以仿真提供的逐帧完整物体点云作为辅助监督，训练时增强位姿和几何感知，部署时仍只需 RGB。人类、机器人、生成和互联网视频实验显示了较强的跨来源适用性。

#### 方法动机分析
核心假设是可迁移的信息是物体运动，而非示范者肢体动作；完整几何监督可改善 RGB 表征。该设计针对动作重定向的具身依赖、插值增强造成的轨迹失真和视觉几何不足。适用边界是刚体操作、单段视频内相机基本固定，并假定执行期间保持刚性抓持。

#### 方法设计详解
输入为单目视频。先用 SAM2 分割，并取最大掩码帧，由 SAM 3D 重建网格；MoGe-3 校准尺度，FoundationPose 跟踪六自由度位姿，再稀疏化、插值和平滑。轨迹被变换到初始物体或静态目标物体坐标系，因而摆脱相机坐标和示范者具身。残差迁移以位置线性插值、旋转 SLERP 构造新起终点基线，再叠加示范偏差：水平残差旋转并缩放，竖直分量和旋转残差保留，从而适配配置同时保持运动形状。仿真筛选抓姿后，通过固定物体—末端变换与逆运动学生成机器人动作。策略采用 ACT，并增加查询和点云头；训练损失为 ACT 目标加双向 Chamfer 距离，监督每帧完整物体点云的位置与形状，推理不输入点云。实验使用 480×640 双相机输入和单张 RTX 5090。

#### 方法对比分析
相较人体重定向，物体轨迹解除示范者具身绑定；相较简单插值或回放，残差迁移显式保持运动轮廓；相较仅位姿辅助监督，完整点云同时约束物体几何。SAM2、SAM 3D、MoGe-3、FoundationPose 与 ACT 是标准组件，主要创新在轨迹适配机制和训练期几何监督，适合从多来源视频迁移到刚体操作。

#### 实验分析（精简版）
四项任务各使用 500 条成功轨迹：残差迁移成功率为 93.95%，高于插值的 48.63% 和回放的 56.15%；形状误差为 0.068，优于 0.148 与 0.105。未见场景中，每任务进行仿真 200 次、真实 30 次，ACT 3D 相比 ACT 平均提高 12.50 和 14.17 个百分点，支持局部泛化收益。局限是真实验证仅四项任务，灵巧手仅有仿真证据，生成视频转换成功率为 9/20。

#### 实用指南
复现需 Isaac Sim、腕部与前方双相机及资产和机器人配置；零样本表示无真实机器人示范或策略微调。迁移时要替换资产、抓姿和运动学设置并重新生成训练数据，尺度校准不可省略。代码、权重、学习率、损失权重及更完整依赖，论文未说明。

#### 总结
核心思想：以物体运动和几何连接视频与动作
1. 校准网格尺度，在物体坐标系提取跨具身运动。
2. 将示范残差迁移至新起终点，保持运动形状。
3. 由筛选抓姿把物体轨迹转换为机器人动作。
4. 用完整点云辅助监督 RGB 策略，实现无三维输入部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36870v1)
- [arXiv](https://arxiv.org/abs/2609.36870v1)

---

<a id='2609.36605v1'></a>
## [RoboChrono: A Real Robot Benchmark for Streaming Task Understanding](https://arxiv.org/abs/2609.36605v1)

**Authors:** Yuzhou Wu, Longteng Fan, Zimeng Li, Yu Wanchan, Ting Zhang, Yiyang Ma, Shihao Li, Wei Ying, Jianbin Qin, Jiajian Jing, Fangwen Chen, Yifan Wu, Zichen Zhang, Ruiqi Yang, Weibin Kong, Yihang Xu, Haoran Liu, Zonghang He, Xuyang Liu, YiFan Xiong, Siteng Huang, Tao Xu, Zhuo Xu, Long Chen, Ruoxiang Li

**Published:** 2026-09-29

**Categories:** cs.RO

**Abstract:**

Understanding ongoing robot manipulation requires models to interpret visual observations in relation to interaction history and task progress. We introduce RoboChrono, a benchmark for streaming task understanding comprising 39 scenarios and 34,713 evaluation instances, constructed from real robot executions and complementary bare-hand human recordings. The benchmark evaluates seven tasks grouped into recognition, alignment, and temporal grounding, covering action understanding and anticipation, visual correspondence, temporal ordering, and action localization. Zero-shot evaluation of 18 vision-language models reveals substantial differences across tasks. GPT-6-Astra achieves 98.3% accuracy on Frame Matching but 68.3% on Frame Ordering, while RynnBrain1.1-122B-A10B exhibits a larger gap, reaching 95.4% and 32.9%, respectively. Input ablations on matched questions with five open-weight models further reveal distinct dependencies on visual evidence: removing visual observations reduces Current Action Recognition accuracy by 22.1 percentage points, whereas Next Action Prediction decreases by only 0.7 points. These findings show that strong visual matching does not consistently coincide with strong temporal ordering, and suggest that next-action prediction can be supported by task and action priors even when visual evidence is unavailable. RoboChrono provides a diagnostic setting for examining these differences, highlighting the need for capability-specific evaluation beyond aggregate scores when assessing task understanding in robot manipulation.

### 论文解读

#### 摘要翻译
作者提出流式任务理解基准RoboChrono，由真实机器人执行和徒手人类录像构成，含39个场景、34,713个评估实例，覆盖识别、对齐、时间定位七项任务。18个视觉语言模型零样本评估显示能力分化：GPT-6-Astra帧匹配准确率98.3%、帧排序68.3%；RynnBrain1.1-122B-A10B分别为95.4%和32.9%。对五个开放权重模型的输入消融发现，移除视觉使当前动作识别下降22.1个百分点，而下一动作预测仅下降0.7点，说明视觉匹配能力不等于时序排序能力。

#### 方法动机分析
完整视频会用未来画面解释早期事件，不符合在线机器人只能观察历史的条件。作者假设“看见什么”与“任务进行到哪里”是不同能力，需在因果观测边界下分别测量；本文贡献是诊断性基准，而非控制策略。

#### 方法设计详解
输入为查询时刻以前的视频前缀和问题，部分预测题附带目标。数据含1,816段成功执行视频、约25小时，覆盖29种夹爪、5种灵巧手和5种徒手场景。人工标注动作与状态的起止区间，再生成当前动作、下一动作、目标条件下一动作、帧匹配、跨视角匹配、帧排序、动作时间定位七类题目。前六项输出选项并以准确率评分，定位输出秒制区间，以预测与真值区间tIoU≥0.5计命中；不得输入未来画面。推理设置全部为零样本，不新增训练目标。

#### 方法对比分析
RoboChrono的创新不是新网络或损失，而是将真实操作、时间标注和受限历史观测统一成七维协议。相较通用视频问答，它拆分视觉对应、时序排序、动作定位与进度预测，并用同步多机位测试跨视角对应。因此适合诊断模型能力边界，但不能直接代表闭环控制性能。

#### 实验分析（精简版）
关键证据是：RynnBrain1.1-122B-A10B的匹配与排序相差62.5个百分点；五模型、312题干的去视觉消融使当前识别降22.1点、下一动作预测仅降0.7点，支持“不同任务依赖不同证据”的判断，但低视觉敏感也可能来自任务先验。数据只含成功执行，失败恢复和统计稳健性仍未充分验证。

#### 实用指南
论文提供代码仓库及Hugging Face数据地址；许可证与链接可用性未说明。复现需同步视角、严格截断未来帧、核查时间区间，并保持提示模板一致；帧采样、分辨率和算力未说明，解析失败率达到50%的运行会排除。迁移到其他机器人或数据集需重建动作标签、时间标注和题目，无须基准专用重训。

#### 总结
核心思想：因果观测拆解时序理解
1. 将操作录像标为动作与状态区间。
2. 按查询时刻截断输入，隔离未来证据。
3. 用七类任务分别测量对应、排序和进度理解。
4. 以去视觉消融区分视觉证据与任务先验。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36605v1)
- [arXiv](https://arxiv.org/abs/2609.36605v1)

---

<a id='2609.36602v1'></a>
## [OTRetarget: Joint Robot and Object Motion Retargeting via Optimal Transport](https://arxiv.org/abs/2609.36602v1)

**Authors:** Guillaume Besset, Erwann Carn, Timothée Carecchio, Valentin Tordjman-Levavasseur, Fabian Schramm, Yann de Mont-Marin, Justin Carpentier, Ajay Suresha Sathya

**Published:** 2026-09-29

**Categories:** cs.RO

**Abstract:**

Transferring human motion to humanoid robots requires adapting the demonstrated motion to the robot morphology while preserving interactions with the environment. This is particularly challenging for loco-manipulation tasks, where contacts with the ground and manipulated objects must remain consistent despite differences in body proportions. Yet, skeletal motion alone does not fully describe these interactions, and fixing object trajectories limits the adaptation to a new embodiment. In this paper, we introduce OTR ETARGET, a unified approach to jointly retarget robot and multi-object motion from human demonstrations. Our approach represents surface interactions through signed distances, closest surface points, and relative directions, and uses entropic optimal transport to transfer these quantities across human, robot, and object geometries. We incorporate the resulting interaction targets into a constrained inverse kinematics formulation that balances contact preservation with motion style and jointly optimizes robot and object poses at each frame. This formulation accommodates robot-object and object-object interactions without rescaling the scene or the demonstration. We validate the proposed approach on OMOMO, where it achieves a robot- object interaction Jaccard score of 87% and a depth error of 8.7 mm, compared with 28% and 29.3 mm for OmniRetarget. Finally, we demonstrate transfer to a physical G1 humanoid using whole-body policies trained with reinforcement learning on the retargeted references, across motions including two-handed box pick-and-place onto a table.

### 论文解读

#### 摘要翻译
OTRETARGET 将人类动作迁移到形态不同的人形机器人，同时保持机器人、物体和环境的接触关系。它以有符号距离、最近表面点和相对方向表示表面交互，用熵正则最优传输在人体、机器人及多个物体几何间建立对应，再通过约束逆运动学逐帧联合优化机器人和物体。方法无需缩放场景或示范，OMOMO 上机器人—物体交互 Jaccard 为87%、深度误差为8.7毫米，并在实体 G1 上展示双手搬箱。

#### 方法动机分析
整体缩放会破坏桌面等固定环境的接触，固定物体轨迹又可能超出新机器人可达范围；仅匹配骨架也不能保证手掌贴合物体。作者的核心假设是迁移表面关系而非复制坐标，并允许物体运动随机器人具身适应。

#### 方法设计详解
输入人体网格、骨架、物体网格及轨迹，输出机器人配置和物体位姿序列。方法密集采样表面，在物体局部坐标中提取距离、最近点、方向三元组；人体部位与机器人连杆配对，点云归一化后用 Sinkhorn 最优传输确定对应。距离残差保持间隙，测地距离保持接触位置，单侧方向惩罚避免贴错表面；示范负距离截为零以避免复制穿透。联合逆运动学同时平衡交互、姿态风格和时间平滑，并约束碰撞、关节及速度。物体替换时再做物体间传输；归一化仅用于匹配，不缩放场景。距离场分辨率为1厘米，正则系数0.1，SQP 首帧最多50次、后续6次。

#### 方法对比分析
相较固定物体并整体缩放的 OmniRetarget，本文以表面交互残差和机器人—多物体联合优化解决可达性与接触保持问题，并能把关系迁移到新物体；与基线对比的核心差异是优化对象从预设物体轨迹变为可调整的交互关系。最优传输、SQP 和强化学习属于标准组件。方法依赖几何与轨迹输入，不是端到端视频控制。

#### 实验分析（精简版）
评估使用2027条 AMASS 与4421条 OMOMO 序列，报告中位数。OTRETARGET 的 Jaccard 87%、深度误差8.7毫米，OmniRetarget 为28%和29.3毫米；固定物体、缩放示范的消融仍有82% Jaccard。OmniRetarget 的424条不可行序列被排除，比较需注意口径。实体搬箱成功6/8次；逐帧运动学尚不保证力学可行性，结果支持接触保持但仍有动力学边界。

#### 实用指南
论文提供项目页，但未说明代码、模型是否开源。复现需重建表面距离场、部位对应，并配置 Pinocchio、ProxQP；下游 PPO 使用4096个环境、30000次迭代，操作任务另加物体速度与力奖励。换机器人须重建几何、限位和对应，并重训跟踪策略。

#### 总结
核心思想：迁移表面交互并联合调物体
1. 将示范接触编码为局部表面三元组。
2. 用最优传输建立跨具身及物体对应。
3. 以交互残差联合优化机器人与物体。
4. 用重定向参考训练全身跟踪策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36602v1)
- [arXiv](https://arxiv.org/abs/2609.36602v1)

---

<a id='2609.36596v1'></a>
## [HACo: Learning Haptic Active Compliance for Force-Aware Dexterous Manipulation](https://arxiv.org/abs/2609.36596v1)

**Authors:** Naisheng Ye, Yinzhe Zhou, Junkai Zhao, Yuhang Lu, Checheng Yu, Zhenjie Yang, Pengwei Wang, Hongyang Li

**Published:** 2026-09-29

**Categories:** cs.RO

**Abstract:**

Contact-rich dexterous manipulation requires policies that translate physical feedback into motion commands while regulating interaction loads across evolving multi-contact interactions. This requires haptic observations of contact state and action supervision showing how commands should adapt. Existing policies often overlook complementary fingertip tactile and joint-torque feedback, while common action targets either encode excessive loading or omit motion constrained by the object. We introduce HACo, a Haptic Active Compliance policy that learns force-regulating actions directly from haptic feedback. Compliance-regulated teleoperation converts operator inputs into controller-executable compliant actions that preserve motion intent while regulating loads. HACo learns these actions directly, using command-state discrepancy as auxiliary compliant-intent supervision. It combines local fingertip tactile responses with joint-torque feedback capturing load transmission through the articulated hand, including contacts beyond tactile coverage. A Compliance Grounding Module uses gated haptic cross-attention to ground action generation in the evolving haptic state, enabling closed-loop force regulation without explicit online contact modeling. We evaluate HACo on a real-world benchmark covering multi-contact friction, tangential interaction, fragile curved-surface contact, rotational torque, and deformable-object manipulation. Across 20 trials per task, HACo achieves an 83% mean success rate, compared with 35% for the strongest evaluated baseline. These results demonstrate active compliance across diverse force-sensitive dexterous manipulation tasks.

### 论文解读

#### 摘要翻译
接触密集型灵巧操作要求策略把物理反馈转成运动指令，并在多点接触变化时调节载荷。现有策略常忽略指尖触觉与关节力矩的互补性：名义遥操作指令可能带来过大载荷，观测构型又会遗漏被物体约束的运动。本文提出 HACo，从触觉反馈直接学习力调节动作。柔顺遥操作把操作者指令转为可执行的柔顺动作，在保留运动意图的同时调节载荷；HACo学习这些动作，并以指令与状态差提供辅助柔顺意图监督。它融合指尖局部接触与关节力矩反映的手部载荷传递，通过门控触觉交叉注意力闭环调力，无需显式在线接触建模。真实多接触任务中，每任务测试20次，平均成功率83%，强基线为35%。

#### 方法动机分析
视觉难以判断压力过大或牵引不足，仅学习实际构型又会丢失维持接触力所需的指令偏移。现有方法或只用指尖触觉，无法观测触觉覆盖外的载荷；或只用关节力矩，难以分辨局部变形、剪切和滑移，直接拼接还会削弱预训练动作表示。核心假设是：经调节的运动参考配合互补物理反馈，能够教会策略调力，而不必直接预测目标力。

#### 方法设计详解
输入为三路图像、语言、机器人状态及触觉历史。示教时，机械臂导纳响应超限力；手部裁剪超限指尖力，经虚拟刚度转为卸载位移，再用雅可比优化得到平滑关节修正。模型先融合同一手指的力矩历史、触觉力和形变图，再跨指建模；动作特征通过零初始化门控交叉注意力查询动态触觉。条件流匹配联合学习柔顺动作与意图，意图为Δq_ci=q_cmp−q_obs，损失权重0.5；推理仅执行柔顺动作。输出40步动作块，以已承诺的10步前缀异步续接。训练基于GR00T N1.7，4张H100、全局批量48、3万步，AdamW学习率2×10^-5。

#### 方法对比分析
不同于先预测接触状态、再映射为控制目标的方法，HACo直接学习调节后的参考；不同于观测构型或名义命令监督，它保留命令—状态差中的受约束运动意图。具体创新包括柔顺动作监督、按手指融合触觉与力矩、以及动作查询触觉的门控交叉注意力；导纳控制和流匹配属于标准组件。它适合需要主动调力的接触任务，但依赖相应传感器与调节示教，不能直接替代无力觉硬件。

#### 实验分析（精简版）
五项双手任务各收集100条示教，每方法每任务测试20次。HACo成功率83%，较T-Rex的35%高48个百分点；仅触觉为68%，仅力矩为45%；移除意图监督降至73%，改用名义动作降至59%，支持互补感知和监督设计。局限是仅单平台、单训练种子，未报告置信区间；成功率也不能完全证明精确力跟踪或跨物体泛化。

#### 实用指南
论文给出项目链接，但代码、权重和数据的开放状态未说明。复现需准备力矩与接触力9步历史、240×240形变图，并保持动作和意图分别归一化。迁移到其他机器人或数据集时，需替换运动链、雅可比和传感器映射并重新采集调节示教；力阈值、控制增益和部署频率论文未说明。

#### 总结
核心思想：以柔顺示教学习触觉调力
1. 调节名义指令，构建可执行柔顺目标。
2. 用指令—状态差监督柔顺意图。
3. 按手指融合触觉与力矩，门控接入动作生成。
4. 闭环执行柔顺动作，辅助意图不直接执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36596v1)
- [arXiv](https://arxiv.org/abs/2609.36596v1)

---

<a id='2609.36588v1'></a>
## [Cooperative Multi-Agent Vision-Language-Action Models via Reinforced Fine Tuning](https://arxiv.org/abs/2609.36588v1)

**Authors:** Ruixiao Xu, Wong Lik Hang Kenny, Zhiqian Liu, Jianing Guo, Hanxiao Li, Kejian Shi, Shuning Zhang, Pu Feng, Yongjia Ma, Yuqing Ma, Kai Chen, Qi Dou, Yaodong Yang, Xianglong Liu, Simin Li

**Published:** 2026-09-29

**Categories:** cs.RO, cs.AI, cs.MA

**Abstract:**

We study reinforcement learning (RL) methods for cooperative multi-agent Vision-Language-Action (VLA) models. This problem is challenging because VLAs are pretrained on large-scale single-agent data and therefore lack the fine-grained coordination skills required for inter-robot collaboration. Supervised fine-tuning (SFT) on multi-robot demonstrations partially bridges this gap, but its performance is bounded by the demonstration data and cannot improve from its own experience. We present a three-stage reinforced fine-tuning (RFT) pipeline for multi-agent VLAs. First, initialization-aware data collection sweeps over initial configurations and invokes human demonstrations only when the pretrained VLA repeatedly fails, yielding robustness to initialization shift with reduced human cost. Second, offline credit-filtered tuning assigns credit to individual agents and fine-tunes on per-agent trajectories with positive advantage rather than on entire joint rollouts. Third, we find existing online RL for VLAs are less effective for hard multi-agent tasks, which we attribute to noisy co-exploration and unstable updates. We instead use online latent-space fine tuning, which freeze the VLA and perform RL in its latent noise space. We evaluate our multi-agent VLA with both $π_0$ and $π_{0.5}$ backbones across 11 tasks in RoboTwin, RoboFactory and real-world manipulation with two Franka robots. Our multi-agent VLA improves the average success rate by $+23.1\%$, $+16.4\%$, and $+44\%$ on RoboTwin, RoboFactory, and real-world tasks, respectively. Code available at https://anonymous.4open.science/r/mavla_rft-2BC0/.

### 论文解读

#### 摘要翻译
本文研究协作式多智能体视觉—语言—动作（VLA）模型的强化学习（RL）。VLA在大规模单智能体数据上预训练，缺乏机器人协作所需的精细协调能力。多机器人示范上的监督微调（SFT）可部分弥合差距，但性能受示范限制，不能从自身经验改进。本文提出三阶段强化微调（RFT）：初始化感知采集遍历初始配置，仅在预训练VLA反复失败时引入人工示范，以较低人工成本增强初始化偏移鲁棒性；离线信用过滤为各智能体分配贡献，仅微调正优势个体轨迹，而非完整联合轨迹；针对现有在线RL在困难多智能体任务中探索噪声大、更新不稳定的问题，冻结VLA，在潜在噪声空间执行RL。在RoboTwin、RoboFactory及双Franka实机共11项任务上，使用π0和π0.5骨干，平均成功率分别提高23.1%、16.4%和44%。论文提供代码链接。

#### 方法动机分析
联合失败不等于所有机器人都做错；独立动作噪声又会破坏同步。核心假设是：补齐失败初始化、保留个体有益动作、限制探索空间，能推动协作学习。有限试验零成功并不能证明技能缺失；理论采用全可观测状态，实际策略使用局部观测。

#### 方法设计详解
输入为腕部图像、本体状态、共享第三视角及指令，输出各机器人动作块。1）每个初始化试运行K次；有成功则保留成功与失败轨迹，否则补人工示范。2）共享Transformer依次接收前m个智能体动作，以同一蒙特卡洛回报监督201桶价值分布。差分 A_m=Q_m−Q_{m−1} 衡量新增动作的贡献；筛选后用标准流匹配微调，实际保留约5%样本，并加入全部人工示范。3）冻结VLA，用DSRL训练小型策略生成初始噪声，再经冻结流模型得到动作。单调改进依赖价值准确、数据支持覆盖及更新后优势仍非负，不是无条件保证。

#### 方法对比分析
相较CHORUS的监督模仿，新增初始化补洞和逐智能体信用筛选；相较Flow-SDE、Flow-Noise，不直接更新VLA，而是学习噪声分布。DSRL、流匹配和优势分解是已有组件，贡献主要在协作场景的组合与验证。适用于已有基础技能、能重置环境并获取示范的任务。

#### 实验分析（精简版）
覆盖8项仿真、3项实机任务，每项分别评估100、50次。四项困难任务中，最终平均成功率69.0%，高于Flow-Noise的61.5%；去掉初始化感知采集后为61.5%。实机Handover由42%升至86%，体现精细交接收益。局限是未报告跨训练种子误差，信用过滤缺少独立对照；附录称实机DSRL使用50次轨迹做离线训练，不能作为持续在线学习证据。

#### 实用指南
[代码链接](https://anonymous.4open.science/r/mavla_rft-2BC0/)由论文提供，模型和数据开放情况未说明。仿真每任务50条SFT示范，扫描96种初始化、各运行5次；微调批量32、训练30000步，DSRL探索200回合。复现须核对正文与附录的失败惩罚−100/−1000及折扣设定差异。迁移需适配相机、本体状态、动作接口，重采示范并重训价值与潜策略；算力未说明。

#### 总结
核心思想：补齐初始化并筛选协作贡献
1. 扫描初始化，仅为零成功配置补示范。
2. 用前缀价值差分筛选个体高贡献动作。
3. 冻结微调后的VLA，以潜噪声策略优化协作。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36588v1)
- [arXiv](https://arxiv.org/abs/2609.36588v1)

---

<a id='2609.36530v1'></a>
## [Trajectory-Level Mode Guidance for Controllable Diffusion-Based Multi-Robot Motion Planning](https://arxiv.org/abs/2609.36530v1)

**Authors:** Tianyou Yu, Shengze Cai, Chao Xu

**Published:** 2026-09-29

**Categories:** cs.RO

**Abstract:**

Motion planning often admits multiple feasible solutions, making multimodal generation valuable, particularly for flexible multi-robot coordination. Diffusion models naturally learn such trajectory distributions, yet incorporating coarse and partial trajectory priors without restricting generation remains challenging. Such priors indicate a desirable region of the solution space rather than a single solution, motivating conditioned generation that preserves multimodality. In this paper, we guide trajectory generation in the clean trajectory space and progressively incorporate trajectory priors with a timestep-dependent guidance strength. At each reverse diffusion step, the reconstructed clean trajectory provides a unified space for integrating planning costs and partial trajectory priors. Planning costs are incorporated through gradient-based refinement, while the partial prior is progressively injected at the corresponding noise levels with decreasing guidance strength. This guides generation toward the prior in early stages while gradually releasing the constraint to preserve the inherent multimodality of the diffusion model. The framework naturally extends to multi-robot planning by incorporating inter-robot collision costs. Experiments on single- and multi-robot planning tasks demonstrate controllable trajectory synthesis, diverse feasible solutions, and safe multi-agent coordination.

### 论文解读

#### 摘要翻译
运动规划常有多个可行解，多模态生成适合灵活的多机器人协作。扩散模型能学习轨迹分布，但粗略、局部先验不应把生成限制为单一解。本文在干净轨迹空间引导，并随时间步减弱先验强度：每个反向步骤重建干净轨迹，用规划代价梯度修正，再在对应噪声水平注入递减先验。生成早期靠近先验，后期释放约束以保留多模态性；加入机器人间碰撞代价即可扩展到多机器人规划。

#### 方法动机分析
粗规划或人类指令可能不完整、错误，硬约束会排除可行解；高噪声轨迹也不适合计算物理代价。核心假设是先验负责早期模式选择，后期由扩散模型与代价恢复多样性和可行性。它是软引导，不能自动保证所有约束满足。

#### 方法设计详解
输入为起终点、环境距离场、可选先验和掩码。轨迹转到局部坐标并按10米归一化；三层编码—解码1D U-Net学习60个二维路点的噪声预测，网络不输入任务条件。采样从高斯噪声开始：先预测噪声并重建干净轨迹，再在掩码处按 (w(τ)=\frac12(e^{2τ}-1)/(e^2-1)) 融合先验；随后对速度、障碍距离、目标、机器人间距和平滑代价做梯度更新，反向转移后再注入匹配噪声水平的先验。关键是干净空间优化与噪声空间注入的双重引导，无需重训条件网络。

#### 方法对比分析
相较条件训练，任务适配发生在采样时；相较直接对噪声态算代价，优化对象有清晰几何意义；相较固定先验，递减权重允许偏离错误先验。扩散、梯度指导和掩码是已有组件，主要创新在组合与调度，适合有粗先验的连续轨迹规划。

#### 实验分析（精简版）
实验使用自建单机器人数据，比较前缀与均匀稀疏条件，未报告外部基线、独立消融或显著性检验。8机器人中，条件路点由6增至30时，平均路径从7.359到6.784米，但最小间距从0.966到0.901米，显示效率与间距权衡。多模态主要由可视化支持，缺少定量多样性指标；10机器人最小间距为0.705米，低于1米设定，安全裕度未被充分证实。

#### 实用指南
训练集含2万条由A*与优化生成的轨迹；Adam训练15万步，批量128、学习率 (10^{-4})。扩散1000步，默认DDIM采样300步，指导步长0.0025。论文未说明数据划分、硬件、耗时和依赖，代码与数据管线承诺发表后公开。迁移需替换距离场、机器人几何及动力学代价，高维状态还需调整模型并重训。

#### 总结
核心思想：先验选模，退火释放约束
1. 重建干净轨迹，定位可指导的几何空间。
2. 掩码注入局部先验，再用规划代价修正。
3. 回到噪声态，注入匹配噪声水平的先验。
4. 逐步减弱引导，以碰撞代价耦合多机器人。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36530v1)
- [arXiv](https://arxiv.org/abs/2609.36530v1)

---

<a id='2609.36844v1'></a>
## [GlassFormer: Learning Real-time Glass Segmentation using Radar-Depth Fusion](https://arxiv.org/abs/2609.36844v1)

**Authors:** Suhani Grover, Astik Srivastava, Viswas Dinesh, Avinash Sharma, K. Madhava Krishna

**Published:** 2026-09-29

**Categories:** cs.CV, cs.RO

**Abstract:**

Transparent surfaces are ubiquitous in built environments, yet they remain a persistent failure case for robotic perception. RGB cameras perceive the background behind glass rather than the surface itself, while depth sensors such as LiDAR, time-of-flight, and RGB-D often return invalid or background measurements in transparent regions. As a result, systems that rely solely on optical sensing may misinterpret glass walls, doors, or mirrors as free space, compromising safe and reliable navigation. Existing glass segmentation approaches address this by learning visual cues such as reflections, boundaries, and semantic context from RGB images. While effective under favourable lighting and viewing conditions, these cues degrade in low-light environments, under glare, or when glass surfaces are featureless or partially occluded. In this work, we propose a multimodal framework that fuses millimetre-wave radar with RGB-D sensing for real-time transparent surface segmentation. Radar reflects strongly off glass surfaces, providing a geometric cue that remains reliable precisely where vision and depth fail. We exploit this cross-modal inconsistency to generate a radar-guided spatial prior, which is integrated into a lightweight transformer-based segmentation network, GlassFormer, via cross-modal attention. We report results on a mixed-condition test split covering all scene types and a dedicated low-light split designed to stress vision-only methods. GlassFormer achieves 0.88 mIoU on the mixed split, and 0.59 mIoU on the low light split, demonstrating substantial robustness gains over vision-only baselines while maintaining real-time performance on resource-constrained platforms.

### 论文解读

#### 摘要翻译
透明表面使RGB看到背景，LiDAR、飞行时间和RGB-D又常返回无效值或背景，机器人可能把玻璃墙、门误判为空闲空间。本文融合毫米波雷达与RGB-D，利用跨模态不一致生成雷达引导先验，并以轻量Transformer网络GlassFormer进行实时分割。作者采集同步数据，覆盖玻璃门窗、镜面及从白昼到近乎黑暗的光照；混合测试和低光测试的mIoU分别为0.88和0.59。这类几何互补旨在让系统在视觉线索退化时仍可识别障碍。

#### 方法动机分析
核心假设是：雷达发现近处界面，而光学深度无效或落在更远背景时，该区域可能透明。雷达不受光照影响却缺少像素级分辨率，故只提供可被网络抑制的候选先验。边界在于雷达主峰可能来自其他物体，错误先验需被抑制。

#### 方法设计详解
输入同步RGB、深度和一维雷达幅值谱。取最大幅值距离并按相机内参近似投影为ROI；在ROI内，将深度无效或与雷达距离差超过阈值的像素标为候选，再用5×5开闭运算去噪。RGB进入SegFormer-B2，候选掩码经两层卷积编码，在后两阶段作为查询、视觉特征作为键和值；可学习空间门控后残差融合，多尺度解码输出分割。训练使用等权BCE与Lovász hinge损失。硬件为XM125与D455，雷达有效距离约6米。推理阶段沿相同路径生成先验并输出像素掩码，方法依赖雷达与相机视场重叠。

#### 方法对比分析
不同于只依靠反射、边界和语义的视觉方法，本文利用传感器失效模式的互补性。相较面向抓取、处理高维雷达的FuseGrasp，本文用一维距离先验服务场景分割。主要新机制是先验构造和门控融合，骨干与损失属于标准组件，适合视觉低光、眩光或无纹理场景。

#### 实验分析（精简版）
数据集含1800帧，比较GDNet、GlassSemNet、SegFormer及独立先验。低光mIoU由SegFormer的0.4912升至0.5904；明亮条件仅由0.8799升至0.8818，说明收益主要来自困难光照。独立先验达到0.5407，说明粗略几何线索不能替代融合网络。RTX-4060报告70.6 fps、CPU报告13.2 fps，但是否含完整预处理未说明。未跨数据集验证，划分比例和部分消融也未说明。

#### 实用指南
论文给出代码链接：https://github.com/Suhani92/GlassFormer；权重和数据开放状态未说明。复现需同步、标定XM125与D455并重做投影；学习率、训练轮数、输入尺寸和阈值未说明。还需检查实时同步误差。迁移到其他平台或任务时需适配传感器几何并重新训练、验证。

#### 总结
核心思想：用跨模态失配定位玻璃
1. 提取雷达最强距离并投影视场。
2. 结合深度缺失与距离差生成透明先验。
3. 以先验查询视觉特征，门控抑制噪声。
4. 融合多尺度特征输出像素级分割。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36844v1)
- [arXiv](https://arxiv.org/abs/2609.36844v1)

---

<a id='2609.36851v1'></a>
## [RoXDrive: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving via Action-Faithful Rollouts](https://arxiv.org/abs/2609.36851v1)

**Authors:** Hongbin Lin, Chaoda Zheng, Yiming Yang, Xiangyu Li, Shijia Chen, Jinhao Deng, Kangjie Chen, Dongbin Zhang, Jie Feng, Yu Zhang, Xianming Liu, Shuguang Cui, Boyang Wang, Zhen Li

**Published:** 2026-09-29

**Categories:** cs.CV

**Abstract:**

End-to-end autonomous driving policies are commonly trained via imitation learning on logged demonstrations without observing the consequences of their own actions, leading to causal confusion in closed-loop real-world deployment. To address this issue, reinforcement learning (RL) post-training offers a promising alternative by leveraging world models as interactive training environments to enable future scene generation for policy improvement. Nevertheless, existing approaches either rely on reconstruction-based simulators, offering limited counterfactual interaction, or adopt synthetic simulators to enable long-horizon closed-loop interaction at the cost of a substantial sim-to-real gap. Recently, video world models have exhibited the ability to generate realistic multi-step future rollouts but may not faithfully reflect action conditions, resulting in action-vision mismatch. In this paper, we introduce RoXDrive, a plug-and-play closed-loop RL framework that enables reliable policy optimization by identifying action-faithful world-model rollouts, consisting of two stages: 1) Model pre-training: In addition to imitation-based policy pre-training, we devise an Action-Vision Faithfulness Evaluator for inverse dynamics estimation with our geometry-aware auxiliary trajectory supervision, enabling long-horizon assessment of whether visual dynamics faithfully reflect the conditioning ego actions. 2) Action-faithful RL post-training: Agents iteratively interact with world models to form long-horizon scene rollouts, retaining only action-faithful ones for dense safety-aware scoring and scene-level closed-loop RL post-training. Extensive experiments on nuScenes and an in-house dataset with over 130K training scenarios demonstrate consistent gains across planners, reducing safety violations by 27.6% with DiffusionDrive on nuScenes and 33.7% with Qwen3-VL on the internal data.

### 论文解读

#### 摘要翻译
端到端自动驾驶策略通常通过日志示范进行模仿学习，无法观察自身动作的后果，导致真实闭环部署中的因果混淆。强化学习（RL）后训练可利用世界模型作为交互环境，生成未来场景以改进策略。然而，重建式仿真器的反事实交互有限，合成仿真器虽支持长时闭环交互，却存在显著仿真到现实差距。视频世界模型能生成逼真的多步未来，但未必忠实响应动作条件，造成动作—视觉失配。本文提出即插即用框架RoXDrive，通过识别动作忠实的世界模型推演实现可靠优化，包含两阶段：1）模型预训练：除模仿学习策略外，以几何感知辅助轨迹监督训练逆动力学动作—视觉忠实度评估器，判断视觉动态是否反映自车动作；2）动作忠实RL后训练：策略迭代交互形成长时推演，仅保留忠实样本，进行密集安全评分及场景级闭环RL。nuScenes和含逾13万训练场景的内部数据实验显示跨规划器提升：DiffusionDrive与Qwen3-VL的安全违规分别减少27.6%和33.7%。摘要称代码位于RoXDrive。

#### 方法动机分析
逼真视频不等于正确动作后果；错误环境反馈可能误导RL。核心假设是：从视频反推运动并核对输入动作，能筛选可信监督。但自车运动一致不保证其他车辆响应正确。这使离线模仿学习难以评估动作后果，也使长时展开中的误差被RL放大；因此筛选器必须同时关注位移、航向和时间累积。

#### 方法设计详解
前视视频→Cosmos3-Nano逆动力学→逐帧相对位姿。除标准Rectified Flow损失，新增多时域位置、航向Smooth-L1监督，约束位姿累积，抑制早期航向误差放大；仅在噪声水平0.2–0.7启用。

策略从轨迹候选中采样→冻结X-World生成多视角未来→策略再次决策。默认同场景生成6条推演；评估器将ADE、FDE和偏航误差归一化取最大值，保留不超过0.75者。

碰撞或越界终止并罚−5，否则奖励进度、舒适及净空。组内回报标准化为优势，加权整段平均动作对数概率；配合KL、熵及困难场景专家修正。训练使用700场景训练集、150场景验证集；推理时每个场景默认生成6条展开。部署无需世界模型。

#### 方法对比分析
区别于合成或重建仿真，本方法利用视频反事实生成；区别于直接世界模型RL，新增忠实度筛选。几何监督是评估器的关键修正；组相对优化本身是已有组件，扩展点在比较同起点闭环后果。

#### 实验分析（精简版）
nuScenes按700/150场景训练/验证，闭环评估含4675个4秒片段、两次动作。DiffusionDrive违规总数由1038降至751，驾驶分数0.526→0.588；不筛选时违规909，支持过滤有效。评估器6秒ADE相较普通微调由1.27降至0.95米，说明几何辅助监督改善忠实度。局限是评估仍依赖X-World，不能等同真实道路收益。

#### 实用指南
需要同步多视角12Hz视频、位姿、对象和地图边界；X-World先适配nuScenes再冻结，忠实样本不足两条则跳过。正文称代码可用，但所给文本无可核验链接；X-World明确闭源，内部数据开放状态未说明。算力和完整训练配置未提供。迁移需重训逆动力学、适配动作接口及安全几何，不能直接套用低频数据。

#### 总结
核心思想：先验动作忠实再优化闭环
1. 用累积几何监督校正逆动力学。
2. 同场景采样闭环未来并核对动作。
3. 筛选可信推演，以安全回报相对优化。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.36851v1)
- [arXiv](https://arxiv.org/abs/2609.36851v1)

---

