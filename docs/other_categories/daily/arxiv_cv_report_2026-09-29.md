time: 20260929

# Arxiv Computer Vision Papers - 2026-09-29

## Table of Contents

1. [TC-ADA: One-Shot Active Domain Adaptation for Semantic Segmentation](#2609.33432v1)
2. [AMBIT: Anticipatory Multimodal Body Recruitment for Bimanual Tracking on a Humanoid](#2609.33484v1)
3. [TacGooseBumps (TacGB): Retrofitting Normal-Only Tactile Sensors with Shear Encoding for Learning Contact-Rich Manipulation](#2609.34006v1)
4. [DeltaSeek: Toward Active Perception in Evolving Construction Environments](#2609.33836v1)
5. [CodeActionBench: Evaluating Agentic Code-as-Policy for Embodied Manipulation](#2609.33807v1)
6. [Test-Time Spatial Reasoning for Robot Manipulation Using Generative Real-to-Sim](#2609.33982v1)
7. [Robot-GST: geometry-aware spatial-temporal robot policy representation and evaluation](#2609.33872v1)
8. [Estimate, Don't Imitate: Reusing Differentiable State-Based Policies for Visuomotor Control](#2609.34018v1)

---

## Papers

<a id='2609.33432v1'></a>
## [TC-ADA: One-Shot Active Domain Adaptation for Semantic Segmentation](https://arxiv.org/abs/2609.33432v1)

**Authors:** Weihao Yan, Yeqiang Qian, Yueyuan Li, Tao Li, Chunxiang Wang, Ming Yang

**Published:** 2026-09-27

**Categories:** cs.CV, cs.RO

**Abstract:**

Manual dense annotation remains a major obstacle to deploying semantic segmentation models in new driving environments. Active domain adaptation (ADA) seeks label-efficient transfer by annotating only a selected portion of the target domain. Existing ADA methods commonly implement this process through multiple rounds of acquisition, annotation, and retraining. We study a practical one-shot image-level setting that selects and densely annotates a fixed target subset in a single round, followed by uninterrupted adaptation. Within this setting, we develop Target-Calibrated Active Domain Adaptation (TC-ADA) as a joint design of complete-image acquisition and target-calibrated adaptation. Stage~1 uses visual representations from a vision foundation model (VFM) together with semantic predictions from a fixed unsupervised domain adaptation model to select representative and informative target images without target annotations. Stage~2 jointly uses labeled source data, labeled target data, and the remaining unlabeled target data, while calibrating source and target supervision under limited target labels. Extensive experiments across five synthetic-to-real and real-to-real driving transfers show consistent improvements over representative ADA baselines. With only 23 to 46 labeled target images on four transfers and 140 on Mapillary, TC-ADA stays within 1.9 mean intersection over union (mIoU) points of target-only full supervision. Code will be available at https://github.com/ywher/TC-ADA.

### 论文解读

#### 摘要翻译
人工密集标注是语义分割模型迁移到新驾驶环境的主要障碍。主动域适应（ADA）只标注目标域中选定的部分图像，但通常需要多轮采集、标注和重训练。本文提出目标校准主动域适应（TC-ADA），研究一次性图像级设置：单轮选择并密集标注固定目标子集，随后持续适应。它结合视觉基础模型（VFM）表示与固定无监督域适应模型的预测来选图，再用少量有标签目标数据校准源、目标监督。五种合成到真实及真实到真实迁移实验均优于代表性ADA基线。

#### 方法动机分析
多轮流程会打断训练和部署，一次性随机选图又可能语义重复；域偏移下的预测不确定性也未必可靠。作者的假设是，空间覆盖能代表目标分布，可信的稀有类别证据能提高信息量，而有限标签需要校准以避免过拟合。该方法针对驾驶语义分割，一次标注并非无需预训练。

#### 方法设计详解
输入是有标签源域与无标签目标图像。先训练并冻结UDA教师。对冻结VFM的四层特征分别做全局池化和2×4网格池化，逐组件归一化后拼接，保留局部小目标信息。R-DKC逐次挑选图像，最大化“与已选集的距离×局部密度权重×语义权重”：密度抑制孤立异常样本，语义项综合类别面积平方根、预测置信度和逆频率，并过滤过小区域，从而同时考虑多样性、代表性和稀有类。排名前K张统一密集标注。适应时组合源监督、直接目标监督、源—目标混合和目标—目标混合；有效样本数决定直接目标监督的衰减，混合总权重固定为2，目标—目标比例由0.5升至0.8，并用EMA教师生成伪标签。论文给出的设定包括1024×1024裁剪、批量2、40k步，适配器学习率10⁻⁴，解码器学习率为其10倍。

#### 方法对比分析
相较迭代ADA，TC-ADA的选样不依赖适应过程中不断更新的模型；相较单轮MADAv2，它把空间覆盖、密度和可信稀有类联合排序，并把采集与监督校准一起设计。VFM、ClassMix和EMA属于标准组件，真正的新机制是R-DKC选样及随标签预算变化的目标监督配方。它适合目标图像可集中标注、部署后不便反复采集的驾驶场景。

#### 实验分析（精简版）
五种迁移在最低标注预算下的平均mIoU为79.39，高于单轮MADAv2的74.28；四种迁移仅标注23–46张目标图像，Mapillary使用140张，与目标域全监督差距不超过1.9点。在ACDC固定适应流程中，R-DKC标注25张得77.66，随机选样为76.58，说明选样有效，但不能把全部提升归因于采集。局限是计时未含UDA训练和人工标注，随机对照也未构成严格配对显著性检验。

#### 实用指南
论文说明代码将发布于GitHub，未确认预训练权重是否开放。复现需冻结DINOv3-B，训练Rein与HRDA，并统一类别映射、图像预算和标注协议；还需重训UDA并重算目标统计。迁移到其他机器人或数据集时，应重选网格与类别权重，检查小目标过滤和域偏移假设。

#### 总结
核心思想：可信覆盖选样校准监督
1. 冻结多层全局—网格表示，保留局部语义。
2. 以密度、多样性和可信稀有类联合排序，一次标注。
3. 衰减小预算直接监督，逐步强化目标域混合。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.33432v1)
- [arXiv](https://arxiv.org/abs/2609.33432v1)

---

<a id='2609.33484v1'></a>
## [AMBIT: Anticipatory Multimodal Body Recruitment for Bimanual Tracking on a Humanoid](https://arxiv.org/abs/2609.33484v1)

**Authors:** Hanlong Li, Sihan Tan, Takeshi Ashizawa, Benjamin Yen, Kazuhiro Nakadai

**Published:** 2026-09-27

**Categories:** cs.RO

**Abstract:**

A humanoid with 5-DoF arms cannot track generic bimanual end-effector trajectories with its arms alone; pelvis and waist motion must be recruited, but which motion, and when, is not uniquely determined. On a Unitree R1 in fixed double support, the set of dynamically valid recruitment strategies (pelvis pose and waist trajectories) for a task is a diverse continuous manifold, and a deterministic regressor trained on it mode-averages into strategies valid only 35% of the time, against 52% for a conditional variational autoencoder (CVAE) and 82% for the best of 16 CVAE samples. We introduce AMBIT: the CVAE proposes strategies from a preview of the commanded trajectory, a non-learned selector filters, ranks and verifies them, and a receding-horizon loop commits to one with hysteresis. The committed strategy is the reference of the same whole-body differential-IK QP a reactive tracker runs, which keeps authority over residual error. On 160 held-out episodes that admit a valid strategy, in full MuJoCo dynamics under a torque controller, AMBIT reaches 85% success at a 3 cm/15 deg tolerance against 74% for the tracker (disjoint confidence intervals) and recruits the body before the arms saturate in 48% of episodes against 35%. Because diversity is preserved, constraints unknown at training time are enforced by selection alone: under five zero-shot shifts AMBIT beats the warm-started tracker on every shift and matches a test-time re-optimisation baseline 17x more expensive. On a Unitree G1, with hyperparameters unchanged, the protocol reproduces the structure of the valid set and widens the gap over the tracker to 0.85 against 0.53. Five selected strategies execute on the externally supported physical R1, distinct in pelvis excursion and tracking the planned end-effector motion to a median of 11 mm by encoder forward kinematics, which establishes kinematic realisability, not balance.

### 论文解读

#### 摘要翻译
5自由度双臂无法独立跟踪任意双手末端轨迹，必须调用骨盆和腰部，但调用方式与时机并不唯一。在固定双脚支撑的Unitree R1上，动力学有效策略构成多样连续流形；确定性回归因模式平均仅35%有效，条件变分自编码器（CVAE）为52%，16次采样择优为82%。因此，AMBIT用轨迹预览条件化CVAE生成策略，经非学习选择器过滤、排序、验证，再由带滞回的滚动时域机制提交。策略作为与反应式跟踪器相同的全身微分IK QP参考，跟踪器保留末端残差修正权。160个存在有效策略的留出任务中，在全身力矩控制的MuJoCo动力学下，3厘米/15°容差成功率为85%，反应式基线为74%，置信区间不重叠；提前调用身体的比例为48%对35%。分布多样性允许仅靠测试时选择满足训练未知约束。五种零样本偏移下均胜过热启动跟踪器，并匹配计算开销高17倍的再优化基线。G1上保持超参数重跑协议，复现有效集结构，成功率差距扩大至0.85对0.53。外部支撑的实体R1执行五条策略，保持不同骨盆位移；编码器正运动学测得末端对计划误差中位数11毫米，仅证明运动学可实现性，不证明平衡。

#### 方法动机分析
双臂任务的12维末端目标常接近奇异，冻结身体时99%的任务超过严格误差容限，因此身体招募不是可选的补丁。反应式 QP 只有手臂饱和后才移动身体，时机过晚。倾斜、下蹲、扭腰和侧移都可能实现同一手部目标，回归器把这些有效解平均后反而落在无效区域。核心假设是：对同一末端预览，整段有效骨盆和腰部轨迹不是少数离散模式，而是连续多模态集合；保留这种分布并提前选择，能在约束下找到慢而稳的招募方式。范围仍限于固定双脚、不迈步且无物体接触的双手跟踪。

#### 方法设计详解
输入是未来双手 SE(3) 轨迹、剩余时长和当前身体状态。离线阶段为每个任务生成41条候选轨迹，进行完整动力学执行并标注有效性，再用最远点采样保留差异化策略。策略包含骨盆位姿与两个腰角，按50毫秒采样并用每坐标16系数 DCT 压缩。条件 CVAE 以16维潜变量生成策略，采用五样本最小重建损失，β=10^-2；训练集为3000条主数据的80%，并以相同输入的 MLP 回归器作基线。运行时每0.5秒采16个样本，先按骨盆禁入区、腰角、障碍间隙等硬约束筛选，再以末端误差、运动平滑、质心裕度和约束代价排序，最多验证6个候选。验证是在未来1秒的手臂 QP 上进行，首个通过者被提交；若新策略代价未改善15%则保留当前策略，切换时0.3秒融合。提交的身体轨迹只是全身微分 IK-QP 的软参考，末端误差仍由跟踪器优先处理；100 Hz 全身力矩 QP 负责接触、动力学和限幅。单次规划约1.5秒，论文闭环评估按零延迟时钟报告，并另测较慢重规划。

#### 方法对比分析
不同于逐姿态生成IK，AMBIT生成时间一致的整段身体策略；不同于直接规定身体轨迹，软执行允许跟踪器修正残差。创新是分布、预览与选择的结合，并非CVAE或QP本身。

#### 实验分析（精简版）
主数据3000任务，80%训练；闭环测试160任务、三种子。严格2厘米/10°容差下，AMBIT为74%，反应式为51%；去预览后轻度容差成功率从无回退版本82%降至56%。但仅评估oracle找到解的任务，生成任务中52%无解。规划耗时1.5秒且仿真暂停等待，尚非实时验证；缺少预览MPC对照。

#### 实用指南
依赖MuJoCo、Pinocchio、OSQP；需生成并动力学验证策略，再去重、DCT编码。代码、权重、数据开源及训练硬件论文未说明。迁移需替换机器人模型、重建数据并重训；G1实验锁定部分关节，不是原生7自由度臂的直接迁移。

#### 总结
核心思想：预览生成多解，验证后软执行
1. 学习整段有效身体策略的条件分布，避免模式平均。
2. 根据未来双手命令采样，以新约束过滤并验证。
3. 滞回提交策略，软参考交由全身控制修正。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.33484v1)
- [arXiv](https://arxiv.org/abs/2609.33484v1)

---

<a id='2609.34006v1'></a>
## [TacGooseBumps (TacGB): Retrofitting Normal-Only Tactile Sensors with Shear Encoding for Learning Contact-Rich Manipulation](https://arxiv.org/abs/2609.34006v1)

**Authors:** Wenjie Li, Binyu Yang, Yuxin Chen, Ambrose Wang, Masayoshi Tomizuka

**Published:** 2026-09-27

**Categories:** cs.RO

**Abstract:**

Contact-rich policies often fail because distinct physical states look alike yet require different actions. Cameras may not reveal whether a connector is aligned or fully seated, while many normal-only tactile sensors can miss the tangential interactions perpendicular to the grasping direction that distinguish these states. We ask whether a learning policy needs calibrated shear measurements, or only a repeatable observation that separates shear-dependent contact states. We introduce TacGooseBumps (TacGB), a passive domed film that mechanically encodes tangential loading as pattern changes in an existing sensor's pressure map. Tangential loading tilts each dome and redistributes pressure across its footprint; an end-to-end policy consumes the resulting maps without added electronics, force reconstruction, or taxel-level dome alignment. Across four imitation-learning tasks and two data-collection pipelines, TacGB improves goal attainment, efficiency, and contact quality: insertion success increases by up to 36 percentage points, and successful insertions are completed faster, while fragile-object placement becomes gentler and drawing becomes more continuous and straight. Signal, stage-wise, failure-mode, and trajectory analyses link these gains to contact regimes in which task-relevant tangential interactions are poorly resolved by vision and normal pressure alone. Together, these results show that shear need not be measured metrically to benefit robot learning; it can instead be mechanically encoded without changing the underlying tactile sensor or the policy's pressure-map input format.

### 论文解读

#### 摘要翻译
接触密集型策略常因不同物理状态看起来相似、却需要不同动作而失败。相机可能无法揭示连接器是否对齐或完全就位，许多仅测法向压力的触觉传感器也会遗漏区分这些状态的、垂直于抓握方向的切向交互。我们探讨：学习策略需要经过标定的剪切测量，还是只需能区分剪切相关接触状态的可重复观测？我们提出 TacGooseBumps（TacGB）：一种被动穹顶薄膜，将切向载荷机械编码为现有传感器压力图的模式变化。切向载荷使穹顶倾斜，重新分配其覆盖区域的压力；端到端策略直接使用压力图，无需新增电子器件、力重建或穹顶与触觉单元逐一对齐。在四项模仿学习任务、两种数据采集流程中，TacGB 改善了目标达成、效率和接触质量：插入成功率最多提高36个百分点，成功插入更快，易碎物体放置更轻柔，绘图更连续、平直。信号、阶段、失败模式及轨迹分析将收益关联到视觉和法向压力难以辨别任务相关切向交互的接触情境。结果表明，剪切无需被定量测量即可帮助机器人学习；机械编码无需改变底层传感器或策略的压力图输入格式。

#### 方法动机分析
核心问题是接触状态混叠：接触面板、进入插槽、抵达终点视觉相近，动作却不同。假设是稳定、可区分的剪切响应已足够指导动作，不必恢复真实力值。目标是改善可观测性，而非制造标定的三轴力传感器。

#### 方法设计详解
输入为图像、压力图和本体状态。跨越多个触觉单元的弹性穹顶受剪切后倾斜，前缘增压、后缘卸载；法向载荷主要改变幅值。这是新机制。

图像经冻结的 DINOv2-S，压力图经卷积编码，再由 Perceiver 式模块融合，与本体特征共同条件化流匹配动作头。图像与触觉使用当前帧及五个控制步前的帧；预测16步动作、执行前8步。策略学习压力模式到动作的映射，不显式解码剪切。论文未说明具体损失公式及优化超参数。

#### 方法对比分析
相较需机械结构与电子器件协同设计、标定力分量的多向皮肤，TacGB 是可贴附的独立机械编码层。相比更换光学触觉硬件，它保留原电子系统与输入形式。创新在观测接口，视觉编码和动作生成属于标准组件。

#### 实验分析（精简版）
固定策略架构与训练日程，各条件采用相同测试初始化，且与示教配置分离。

1. USB 每条件100条示教、20次测试：裸阵列成功率75%，商品穹顶95%，自制穹顶90%；对应纯视觉策略为60%、65%、55%，支持收益主要来自编码观测，而非表面机械变化。
2. 鸡蛋放置两条件均25/25成功，但 TacGB 将外部杯底传感器平均峰值从0.77降至0.26，支持质量收益；该读数不是标定力值。

优势是低侵入；局限是缺少独立训练种子，不能将重复部署等同于训练稳定性证据。

#### 实用指南
代码、权重和数据开源状态论文未说明。复现可用32×32 Tachin 阵列搭配直径6毫米、间距9毫米的商品穹顶；无需逐单元对齐。需同步视觉与触觉，并保持各条件评估一致。学习率、训练轮数和算力论文未说明。迁移时须匹配覆盖区域、单元间距和动作表示；更换材料或几何可能需要新示教并重训，不能假定零样本适配。

#### 总结
核心思想：以机械编码改善接触可观测性
1. 在法向阵列上覆盖跨单元弹性穹顶。
2. 将切向载荷转成方向性压力重分布。
3. 融合压力时序、视觉与本体状态，直接学习动作。
4. 利用编码变化识别接触转换并调节释放与持续接触。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.34006v1)
- [arXiv](https://arxiv.org/abs/2609.34006v1)

---

<a id='2609.33836v1'></a>
## [DeltaSeek: Toward Active Perception in Evolving Construction Environments](https://arxiv.org/abs/2609.33836v1)

**Authors:** Sanjay Acharjee, Md Nazmus Sakib

**Published:** 2026-09-27

**Categories:** cs.RO, cs.CV

**Abstract:**

Construction environments evolve continuously, causing large geometric changes that degrade static mapping and registration performance. This necessitates active perception, where robots deliberately select sensing configurations to resolve the environment's current state. We present DeltaSeek, an initial framework toward active perception in evolving built environments. While our broader objective is a system that reasons about where, how, and when to observe, this paper addresses a critical prerequisite: how a robot's sensing embodiment constrains the observations it can acquire. We formalize an embodiment's permissible observation set and evaluate with a Husky A300 equipped with a UR5e on an IFC-derived benchmark under chassis-mounted and wrist-mounted RGB-D configurations, scoring observations by geometric visibility and effort by drivable distance. In a room-scale scene with eight controlled changes spanning four observability conditions, exhaustive evaluation over 240 permissible base poses and five arm postures shows that two changes admit no chassis viewpoint whatsoever, while the wrist camera resolves both. For changes observed by both embodiments, the median base travel is $6.0$~m for the wrist camera and $15.2$~m for the chassis camera. These results distinguish sensing limitations from acquisition costs, clarifying whether an observation is impossible or simply requires more travel.

### 论文解读

#### 摘要翻译
施工环境持续演变，大幅几何变化使静态建图与配准性能下降，因此需要主动感知：机器人主动选择感知配置，确定环境当前状态。我们提出 DeltaSeek，一个面向演变建筑环境的主动感知初步框架。长期目标是推理在哪里、如何及何时观察；本文研究其前提：机器人感知具身配置如何约束可获取的观测。我们形式化定义允许观测集合，在 IFC 衍生基准中，以配备 UR5e 的 Husky A300 比较底盘与腕部 RGB-D 配置，用几何可见性评价观测，用可行驶距离衡量代价。在含四类可观测条件、八处受控变化的房间场景中，穷举240个底盘位姿和五种机械臂姿态，发现两处变化不存在可用底盘视点，但腕部相机均能观察。对两者均观察到的变化，腕部与底盘的中位行驶距离分别为6.0米和15.2米。结果区分了感知限制与获取成本，明确观测究竟不可能，还是需要更多移动。

#### 方法动机分析
覆盖扫描不按变化验证需求选视点，大变化又削弱配准对应。本文先问“能否看见”，再问“走多远”，避免将硬件盲区误判为规划失败。假设识别与定位完美，仅研究几何可观测性，不实现完整主动感知闭环。

#### 方法设计详解
输入 IFC 先验、预定义变化和机器人配置。构件以局部有向包围盒近似，裁成7.0×12.5米房间并添加遮挡物。60个位置各取四个朝向，腕部再组合五种姿态，得到240/1200个视点。支持表面采样经视锥、量程、入射角和遮挡筛选，得到可见比例 (V)。定义 (Q_e^s={q:V(e,q)ge0.05})：空集表示该配置空间内不可观测；非空则以到集合的最短可行驶距离定义理想成本。Dijkstra计算距离，贪心规划按单位距离的预测覆盖增益选择视点；输出可见集合及首次观测路程，不更新传感器后验。

#### 方法对比分析
区别于被动验证和面向未知空间的下一最佳视点，本文利用建筑先验研究变化的可达观测集合。创新在“能力—成本”分解和可解释基准；运动学、几何筛选与路径搜索属于标准组件，并无新检测网络。

#### 实验分析（精简版）
单房间八处变化，每类两处；相同相机比较安装方式，无学习基线或系统消融。25米预算内，底盘观察5/8，腕部8/8。两处高度变化穷举后仍无底盘视点；漏掉的入射角变化却有14/240个可用视点，支持能力与路由失败的区分。共同观察的五处，中位路程改善约2.5倍，但这是路线结果，非最优成本证明；高度目标仅超出底盘边界11.8毫米，结论对参数敏感。

#### 实用指南
论文提供[代码与数据链接](https://github.com/iSET-LAB/deltaseek)，未核验内容。无需训练；推理按预设视点穷举与几何可见性筛选；复现需 IFC/IfcOpenShell、机器人运动学及占据栅格。两相机视场86°×57°、轴向深度5米、入射角上限75°。评估不计转向、机械臂运动与返程。迁移需重算相机外参、足迹和视点集合；真实检测及闭环停止规则尚待实现。

#### 总结
核心思想：分离观测能力与获取成本
1. 将变化表面映射到具身配置约束下的视点空间。
2. 用几何可见比例构造目标可观测集合。
3. 区分空集盲区与非空但未访问的视点。
4. 用可行驶距离量化可观测目标的获取代价。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.33836v1)
- [arXiv](https://arxiv.org/abs/2609.33836v1)

---

<a id='2609.33807v1'></a>
## [CodeActionBench: Evaluating Agentic Code-as-Policy for Embodied Manipulation](https://arxiv.org/abs/2609.33807v1)

**Authors:** Yiheng Lyu, Xueying Jiang, Wenhao Li, Shijian Lu, Gongjie Zhang

**Published:** 2026-09-27

**Categories:** cs.RO, cs.AI

**Abstract:**

How well can general-purpose multimodal models turn visual understanding and reasoning into embodied manipulation via executable code? We introduce CodeActionBench, a benchmark of 25 manipulation tasks that evaluates this capability through agentic Code-as-Policy. Without task-specific fine-tuning, demonstrations, external specialist perception or grasp modules, privileged scene state, or predefined task policies, agents should select visual evidence, form task-relevant 3D estimates, construct manipulation targets, and iteratively execute and revise their policies. A shared robot API provides RGB observations, calibrated geometric operations, robot feedback, and bounded motion, leaving task-dependent decisions to the evaluated agent. Fixed task instances, resource budgets, and a hidden physical-outcome verifier support controlled comparisons across models and harness configurations. Extensive evaluations across nine configurations and 675 attempts achieve success rates ranging from 2.7% to 73.3%. The strongest configuration, GPT-6 Astra with Codex CLI, solves 22 of 25 tasks at least once in three attempts, demonstrating the best performance while still leaving substantial room for improvement. Trajectory analyses reveal difficulties in spatial alignment, object retention, and completion judgment, including task failures despite successfully completed motions. CodeActionBench provides a controlled testbed for measuring how general-purpose models translate their capabilities into manipulation behavior and for examining typical failure scenarios in that process.

### 论文解读

#### 摘要翻译
通用多模态模型能否通过可执行代码，将视觉理解和推理转化为具身操作？我们提出 CodeActionBench，以25项操作任务评估智能体式 Code-as-Policy。无需任务专项微调、示范、外部专业感知或抓取模块、特权场景状态或预定义任务策略，智能体应选择视觉证据，形成任务相关三维估计，构造操作目标，并迭代执行、修订策略。共享机器人API提供RGB观测、标定几何运算、机器人反馈及有界运动，任务相关决策留给智能体。固定任务实例、资源预算及隐藏物理结果验证器支持模型与运行框架间的受控比较。九种配置、675次尝试的成功率为2.7%–73.3%。最强配置GPT-6 Astra配合Codex CLI，在每项三次尝试中至少一次解决25项中的22项，表现最佳但仍有显著提升空间。轨迹分析揭示空间对齐、物体保持与完成判断的困难，包括动作成功执行但任务失败。该基准为衡量通用模型如何将能力转化为操作行为、分析典型失败提供受控平台。

#### 方法动机分析
现有代码策略常借助物体坐标或抓取专家，难以区分模型自身能力与外部辅助。本文核心假设是：固定底层执行设施、撤去任务语义辅助，才能检验模型能否贯通视觉证据、空间估计和操作决策。边界是仿真内能力评估，而非新控制器或泛化证明。

#### 方法设计详解
输入任务目标、多视角RGB、相机标定及机器人反馈；模型选择对应像素或平面假设，几何工具计算三维位置；模型再构造抓取方向、TCP位姿和Python策略，执行后根据实际反馈修订。工具只计算，不选择对应点或抓取目标；逆运动学、规划和低层控制属于标准设施。无训练损失。推理阶段每次`run_code`提交计一次调用，内部动作仍消耗仿真时间。调用预算为专家估计分钟数乘14后向上取整；物理预算为示范时长五倍与60秒的较大值，向上取整至5秒倍数。终止后隐藏验证器检查状态或事件，独立比较智能体完成声明。

#### 方法对比分析
相较VLCP提供物体位姿、OpenETA插件通过深度定位，本文让模型承担RGB证据选择、三维估计及目标构造。创新主要是信息边界、统一评估协议及诊断体系，而非几何算法。空间检查点与任务子目标分开统计，避免把“到达位置”误当“完成操作”。

#### 实验分析（精简版）
25个固定场景，每配置75次尝试。Astra＋Codex成功55/75（73.3%），Opus＋参考框架37/75（49.3%）；跨框架差异不能归因于模型本身。Astra空间覆盖87.3%，子目标覆盖76.0%，支持位置到达不等于任务完成。论文未提供因果消融；单场景、三次重复及无实机测试限制泛化结论。

#### 实用指南
代码与日志仅承诺发布，未确认已开源。复现需SAPIEN、RoboTwin 2.0、ALOHA-AgileX和cuRobo；关闭场景随机化，使用固定种子，不提供深度或物体真值。七种参考配置采用提供商默认采样、无固定模型种子。迁移需重建标定、控制接口、预算和验证条件；是否需要重训，论文未说明。

#### 总结
核心思想：隔离辅助评估视觉代码操作
1. 限定RGB与几何证据，隐藏任务特权信息。
2. 模型选证据、估三维并生成操作代码。
3. 根据执行反馈修订策略，自主声明完成。
4. 隐藏验证器联合轨迹诊断区分到位与成功。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.33807v1)
- [arXiv](https://arxiv.org/abs/2609.33807v1)

---

<a id='2609.33982v1'></a>
## [Test-Time Spatial Reasoning for Robot Manipulation Using Generative Real-to-Sim](https://arxiv.org/abs/2609.33982v1)

**Authors:** Ivan Kapelyukh, Yafei Hu, Ran Gong, Brandon May, Tushar Kusnur, Laura Herlant, Karl Schmeckpeper, Edward Johns, Xiaohan Zhang

**Published:** 2026-09-27

**Categories:** cs.RO, cs.CV

**Abstract:**

Spatial reasoning is fundamental to general robot intelligence, as it enables robots to complete long-horizon tasks involving multi-object interaction. We introduce Simify, a training-free, test-time framework that performs explicit spatial reasoning via massively parallel physics simulation. From a single RGB-D image of a scene, Simify reconstructs simulation-ready assets leveraging 3D generative models and vision-language models. Then given a task specified by a reward function (e.g., build the tallest tower), Simify launches thousands of parallel rollouts in simulation and performs an evolutionary search to optimize object arrangements, typically converging within seconds. We conduct quantitative experiments on real-robot hardware to demonstrate the ability of our framework to execute complex object rearrangement tasks end-to-end with previously unseen objects. Results show that our framework outperforms prior work on foundation models for spatial reasoning by effectively exploiting large-scale parallel simulation during inference, and also highlight the importance of complete and accurate geometry for successful sim-to-real transfer.

### 论文解读

#### 摘要翻译
空间推理是通用机器人智能的基础，使机器人能够完成涉及多物体交互的长时程任务。本文提出 Simify，一种无需训练、通过大规模并行物理仿真实现显式空间推理的测试时框架。它从单张 RGB-D 图像出发，利用三维生成模型和视觉语言模型重建可仿真的资产。给定奖励函数指定的任务，例如搭建最高塔，Simify 并行运行数千次仿真，通过进化搜索优化物体排列，通常在数秒内收敛。作者在真实机器人上开展定量实验，展示了使用未见物体端到端执行复杂重排任务的能力。结果表明，该框架通过推理时的大规模并行仿真，优于已有空间推理基础模型方法，并凸显完整、准确的几何对于仿真到现实迁移的重要性。

#### 方法动机分析
语义合理不等于物理可行：堆叠、悬臂和平衡需要精确接触与稳定性判断。核心假设是完整几何与物理搜索可替代任务专用示范，发现新排列。边界是刚体、给定奖励及质量和摩擦参数；语言到奖励的转换不在本文范围内。

#### 方法设计详解
输入单视角 RGB-D 与奖励，输出有序目标位姿。
- SAM-2 提议分割，GPT-4o 合并为物体掩码；二维补全消除遮挡，Hunyuan3D-2 生成完整网格。深度与 ICP 确定尺度，Foundation Pose 估姿，再加载 Isaac Lab。
- 随机初始化放置顺序与位姿；依次放置、等待稳定并计算奖励。保留前 5%，其余按奖励 softmax 采样，再变异顺序和位姿。
- 搜索时加入高斯放置噪声，最终用多次带噪滚动的平均奖励选优，即最大化“执行并稳定后”的期望收益，而非单次幸运得分。
- 执行时重估物体位姿，采样抓取并用 cuRobo 规划；末段力反馈确认接触后释放。

无需任务专用训练，但依赖预训练模型。

#### 方法对比分析
LLM-GROP 直接推断目标位姿，Simify 则用动力学结果修正候选；TSDF-EA 使用相同搜索，但仅重建可见表面；ZeroBot 学习到达给定目标，本文搜索目标本身。贡献主要是单视角几何补全、混合顺序—位姿搜索与鲁棒验证的系统组合，并非新基础模型或新进化算子。

#### 实验分析（精简版）
堆叠、悬臂、骨牌各重复重建与规划 10 次，每个解仿真评估 2048 次；硬件每任务 10 次。
1. 完整方法平均归一化得分 87.5%，去掉最终验证为 64.0%，随机排列为 10.3%，支持搜索与鲁棒选优的作用。
2. YCB 高遮挡网格平均 F1 从 TSDF 的 0.579 升至 0.825，但表面准确性距离由 0.002 升至 0.007，说明补全有精度代价。

各方法使用自身网格评估，并非统一真实几何；搜索也不模拟机械臂，故高分不保证真实可执行。

#### 实用指南
默认 1024 个并行环境，优化平均 73 秒；摘要“数秒”不能视为完整流程耗时。网格生成 octree 分辨率为 64；硬件为 Franka Panda、D435 与力传感器。复现须对齐奖励、轴约束、物理参数和噪声。种群代数、验证次数、噪声尺度及 GPU 型号论文未说明；代码和数据开放状态亦未说明。迁移需替换机器人运动学、抓取与奖励，无须任务专用重训。

#### 总结
核心思想：补全几何后仿真搜索稳健排列
1. 从单视角补全遮挡，构建物理场景。
2. 联合进化放置顺序与连续位姿。
3. 用带噪验证筛选高期望收益方案。
4. 通过重估姿态和接触反馈执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.33982v1)
- [arXiv](https://arxiv.org/abs/2609.33982v1)

---

<a id='2609.33872v1'></a>
## [Robot-GST: geometry-aware spatial-temporal robot policy representation and evaluation](https://arxiv.org/abs/2609.33872v1)

**Authors:** Sichao Liu, Zekun Wang, Lixuan Tang, Yiming Li, Xiaohan Wang, Hanzhi Zhang, Daqiang Guo, Peng Zhou, Lihui Wang

**Published:** 2026-09-27

**Categories:** cs.RO, cs.AI, cs.LG

**Abstract:**

Robotic manipulation policies are advancing rapidly with increasing reliance on vision-language models for end-to-end decision making. However, reliable deployment remains challenging because many policies lack explicit mechanisms for predicting task outcomes and evaluating whether generated actions will achieve desired final states, causing execution errors to accumulate during long-horizon manipulation. We present Robot-GST, a geometry-aware spatio-temporal behaviour representation and evaluation framework that constructs a Gaussian-SAM robotic environment for real-to-sim policy verification and improves the reliability of real-world manipulation deployment. Our approach constructs a high-fidelity robotic environment from RGB-D observations using 3D Gaussian Splatting and SAM3D, enabling ``simulation and evaluation before acting''. It integrates visual observations and language instructions with spatio-temporal reasoning for long-horizon task planning using large vision-language models. To bridge high-level planning and real-world execution, we introduce Gaussian-aware final-state estimation through geometric sampling and state-based trajectory planning. Before execution, candidate action sequences are simulated and evaluated in the Gaussian-SAM environment to filter infeasible behaviours. We validate our approach on representative manipulation tasks involving rigid, soft, and deformable objects, including cube placing, toy packing, and duck rearrangement, demonstrating that geometry-aware spatio-temporal reasoning and state-aware execution improve manipulation reliability across different object categories. Our results suggest that combining geometry-aware reconstruction with high-quality rendering and simulation provides a scalable approach for evaluating robotic manipulation behaviours. Website: https://robot-gst.github.io

### 论文解读

#### 摘要翻译
机器人操作策略正快速发展，越来越依赖视觉语言模型进行端到端决策。然而，许多策略缺少预测任务结果、评估动作能否达到目标终态的显式机制，导致长时程操作中执行误差累积，可靠部署仍具挑战。我们提出Robot-GST：一种几何感知的时空行为表示与评估框架，通过构建Gaussian-SAM机器人环境，开展真实到仿真的策略验证，提高真实操作部署可靠性。方法利用3D Gaussian Splatting和SAM3D从RGB-D观测构建高保真环境，实现“行动前仿真与评估”；结合视觉观测、语言指令和大型视觉语言模型的时空推理，规划长时程任务。为连接高层规划与真实执行，我们通过几何采样和基于状态的轨迹规划，引入高斯感知终态估计。执行前，在Gaussian-SAM环境中仿真并评估候选动作序列，过滤不可行行为。刚性、柔性和可变形物体任务，包括方块放置、玩具装箱和鸭子重排，验证了几何感知时空推理与状态感知执行能提高不同物体类别的操作可靠性。结果表明，结合几何感知重建、高质量渲染和仿真，为机器人操作行为评估提供了可扩展路径。

#### 方法动机分析
稀疏关键点难以描述完整形状、碰撞与变形；直接执行容易积累误差。长时程任务中，前一步的微小位姿偏差会改变后续接触关系，单纯依靠语言规划难以及时发现失败。核心假设是：尺度对齐的几何与近似物理足以提前筛除失败动作，但仿真成功不保证真实成功。

#### 方法设计详解
RGB-D多视角扫描→3DGS外观表示与SAM3D物体网格→统一机器人坐标系。重建损失结合颜色、深度和语义特征误差；高斯负责渲染，网格/SDF负责碰撞，PhysTwin负责软体接触。

GPT-5读取标注图像和指令，生成关键点及分阶段目标、路径约束。抓取后固定物体—末端相对变换，在目标位姿附近采样平移和旋转扰动；用SDF安全距离过滤候选，按姿态稳定性启发式排序并规划轨迹。逆运动学、碰撞及终态谓词均通过才执行；真实失败后重置并重规划，而非保证在线恢复。

#### 方法对比分析
相较ReKep/MOKA，增加完整物体几何、终态采样与执行前验证；相较感知型3DGS和PhysTwin，连接语言规划与行为筛选。贡献主要是系统整合，并非新基础模型；适合可扫描、可建模接触的装箱任务。

#### 实验分析（精简版）
UR5上三类任务各报告10次试验并改变物体位置，无公开数据划分。重规划使真实平均成功率由55.3%升至80.0%，提高24.7个百分点；仿真为66.7%→93.3%。这是相对提升约44.6%，不是增加44.67个百分点。

重建SSIM为0.963、LPIPS为0.044，优于所列基线；引言误将0.044写作SSIM。缺少模块消融、强策略对照和置信区间；鸭子46%的统计口径与10次整任务试验不直接吻合，证据不足以支持“近乎完美部署”。

#### 实用指南
使用D435/D405、RTX 4090；推理阶段在该GPU上完成重建、候选评估与轨迹规划；需标定RGB-D、重建物体、以ICP对齐URDF。采样阈值、损失权重和耗时论文未说明，DINO版本表述不一致。[项目网站](https://robot-gst.github.io)已提供，代码、权重及数据开放状态未说明。迁移需替换机器人模型、标定与终态判据，重新重建并校准物理；是否重训未说明。

#### 总结
核心思想：以几何终态验证约束执行
1. 构建对齐的高斯外观与物理几何双表示。
2. 将语言分解为关键点时空约束。
3. 采样终态并筛除碰撞候选。
4. 仿真验证后执行，失败则重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.33872v1)
- [arXiv](https://arxiv.org/abs/2609.33872v1)

---

<a id='2609.34018v1'></a>
## [Estimate, Don't Imitate: Reusing Differentiable State-Based Policies for Visuomotor Control](https://arxiv.org/abs/2609.34018v1)

**Authors:** Denis Shcherba, Adrian Abel, Eckart Cobo-Briesewitz, Wojciech Samek, Marc Toussaint

**Published:** 2026-09-27

**Categories:** cs.RO, cs.LG

**Abstract:**

Simulation-trained manipulation policies can exploit privileged state information to learn effective contact-rich behaviours, but deployment requires acting from partial observations such as noisy camera images. A common solution is teacher-student distillation, in which a visuomotor policy is trained to reproduce the actions of the privileged expert. This requires the student to jointly infer the task-relevant state and relearn the expert's action mapping that is already available. An alternative is to reuse the state-based expert and learn only a perceptual interface that reconstructs its missing state inputs. However, minimising the state estimate error alone does not necessarily minimise the downstream control error induced by these estimates. To bridge this gap, we train a visual state estimator using both direct state supervision and an action-consistency loss backpropagated through the frozen, differentiable expert. A scheduled objective first establishes a physically meaningful state estimate and progressively emphasises errors that affect the expert's actions. Across five goal-conditioned manipulation tasks, retaining the expert consistently outperforms direct pixel-to-action imitation from the same expert demonstration corpus. We further demonstrate sim-to-real transfer on a physical Panda robot, achieving 76% success without retraining the underlying expert.

### 论文解读

#### 摘要翻译
仿真训练的操作策略可利用特权状态学习有效的接触密集行为，但部署时只能获得含噪图像等部分观测。常见的教师—学生蒸馏训练视觉运动策略复现专家动作，要求学生同时推断任务状态，并重学已有的专家动作映射。另一种方案是保留状态专家，仅学习重建缺失输入的感知接口。然而，最小化状态估计误差不一定最小化下游控制误差。为弥合这一差距，本文结合直接状态监督与经冻结、可微专家反向传播的动作一致性损失训练视觉估计器。调度目标先建立物理上有意义的估计，再逐渐强调影响专家动作的误差。在五项目标条件操作任务中，保留专家始终优于使用同一专家示范库的直接像素到动作模仿。真实 Panda 机器人实现了仿真到现实迁移，无需重训专家，成功率为76%。

#### 方法动机分析
核心假设是：已有控制能力不必重新模仿，但感知精度应按控制敏感性分配。单纯状态回归忽略误差的行为后果，纯动作监督又可能从错误状态出发陷入优化失败。边界是已有可微专家、训练时可获得真实状态，且视觉历史足以估计缺失量。

#### 方法设计详解
输入为四帧多视角图像、同期本体感知和目标。共享 DINOv3 经 LoRA 适配，空间 softmax 提取图像表示；与本体感知拼接后，TCN 聚合时间信息，目标条件 MLP 预测物体及目标相对状态。预测值解码后与实测本体状态拼接，由冻结专家输出关节参考位置增量。

状态损失 (L_s) 是按示范标准差归一化的均方误差；动作损失为 (L_a=\|\pi^*(\hat s)-a^*\|^2)。默认目标为 ((1-p)L_s+pL_a)，其中 (p) 是训练进度；梯度穿过专家，但只更新估计器。局部上，动作损失以 (J^\top J) 加权状态误差，强调控制敏感方向。目标相对量直接预测，未强制几何一致性。

#### 方法对比分析
区别于 ACT、流匹配等重学动作映射，也区别于仅回归物理状态的模块化控制，以及重建专家潜变量的适配方法。主要新增机制是物理状态接口上的渐进双损失；视觉骨干和时序网络属于标准组件。适用于专家可靠、重新学习控制代价较高的任务。

#### 实验分析（精简版）
五项任务各收集一万集示范；主要方法训练三种子，每次评估一千集。AllegroCube 状态估计达79.2%，像素流策略仅0.7%，真实状态流策略为0.4%，支持保留专家。TrayPlate 线性调度将状态监督的28.9%提高到64.1%，但 AllegroCube 反降至77.2%，收益并不普遍。

比较限于固定预算，部分基线仅单种子；双方过滤和正则不同。实机成功19/25，若计入两次安全中止，则为19/27，即70.4%。

#### 实用指南
训练四万步，初始学习率 (2\times10^{-4})，余弦衰减、混合精度；专家使用 MuJoCo Warp 与 TD7/HER。估计器使用全部示范，模仿基线仅成功轨迹。复现须保持状态顺序、归一化及相机标定一致；迁移需重训感知接口，任务或机器人改变时还需匹配专家。代码、权重、数据开源及计算资源论文未说明。

#### 总结
核心思想：保留控制，渐进学习感知接口
1. 将专家输入拆成实测量与待估状态。
2. 用状态监督建立物理可信的视觉接口。
3. 渐增穿过冻结专家的动作一致性梯度。
4. 将估计状态交回原专家闭环控制。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.34018v1)
- [arXiv](https://arxiv.org/abs/2609.34018v1)

---

