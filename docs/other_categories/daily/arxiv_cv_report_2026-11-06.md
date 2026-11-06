time: 20261106

# Arxiv Computer Vision Papers - 2026-11-06

## Executive Summary

## 今日研究脉络

本次十篇论文集中在把视觉模型接入可执行系统时，如何表示不确定性、约束成本并验证闭环效果。自动驾驶与机器人方向既有ZipDrift通过多候选代价修正改进短视MPC，也有CoSafe在RSU端协调多车安全轨迹；MotoVerse与WayPOP则补充两轮交通预测和未知物体全景跟踪的数据评估。它们共同提醒：离线分数需要和交互时序、身份关联及安全失败模式一起阅读。

世界模型方向出现两种轻量化补充：Time-to-Goal从离线轨迹时间关系学习目标距离，MKD2World以空间结构化文本预测网页操作后的界面。前者在四种模拟规划任务中提升成功率，后者在若干GUI骨干上改善网页任务表现，但收益并非处处一致，且动态网页仍未覆盖。METRIC-GS通过置信度路由在廉价尺度估计与重型几何模型间折中；STARE把OCR文字加入机器人对象检索，呈现检索速度与端到端成功率间的差别。

较适合优先精读的是CoSafe（多车交互安全与消融）、ZipDrift（生成候选与约束优化器的分工）、MKD2World（状态表示和动作后果预测）、Time-to-Goal（从无奖励离线轨迹学习规划度量）。ME2FDiff则展示潜空间扩散和多尺度专家路由的医学分割组合。整体趋势是从单一模型得分转向模块化决策、选择性计算和面向边界条件的评测；多数证据仍来自有限基准或模拟环境，真实部署需要独立验证。

---

## Table of Contents

1. [CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving](#2611.06204v1)
2. [ZipDrift: Generating Multimodal Short-Horizon MPC Costs with One-Step Drifting Models](#2611.05793v1)
3. [WayPOP: A Panoramic Open-Set Panoptic Tracking Benchmark](#2611.05801v1)
4. [Time-to-Goal: Learned Temporal Metrics for Long-Horizon Planning with JEPA World Models](#2611.05847v1)
5. [MotoVerse: A Benchmark for Predictive Perception of Motorized Two-Wheelers in Dense Unstructured Traffic](#2611.06007v1)
6. [METRIC-GS: Selective Metric-Scale Estimation for 3D Gaussian Splatting](#2611.06020v1)
7. [STARE: Scene Text-Aware multimodal Retrieval for Everyday objects](#2611.06089v1)
8. [Can Astra Solve RoboMME without Breaking the Bank?](#2611.06184v1)
9. [MKD2World: A Typed Spatial MarkDown World Model For Web Agents](#2611.06544v1)
10. [ME^2FDiff: Multiscale Expert Enhancing Feature Diffusion in Residual Network for Brain Tumor Segmentation](#2611.05766v1)

---

## Papers

<a id='2611.06204v1'></a>
## [CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving](https://arxiv.org/abs/2611.06204v1)

**Authors:** Dingji Wang, You Lu, Haowen Jiang, Yanggang Chen, Yuyang Bi, Bihuan Chen, Shenglin Bao, Xin Peng

**Published:** 2026-11-04

**Categories:** cs.SE, cs.AI, cs.RO

**Abstract:**

With the rapid development of autonomous driving systems (ADSes), real-world roads are evolving into mixed-traffic environments where autonomous vehicles (AVs) and other traffic participants (e.g., human-driven vehicles (HDVs), cyclists, and pedestrians) coexist and interact with each other. However, current ADSes still incur various safety violations in mixed traffic scenarios, raising a strong demand for runtime safety assurance frameworks that can continuously monitor hazards and prevent potential violations. Existing ego-centric assurance approaches are insufficient for mixed traffic, where violations often arise from locally reasonable yet jointly incompatible decisions among multiple vehicles and cannot be reliably resolved through independent takeover.   To bridge this gap, we propose CoSafe, a modular and ADS-agnostic cooperative runtime safety assurance framework for mixed-traffic autonomous driving to extend runtime safety assurance from ego-centric takeover to scene-level cooperative intervention. CoSafe leverages vehicle-to-everything (V2X) communication to enable roadside units (RSUs) to monitor scene-level hazards and jointly mitigate hazards involving multiple vehicles. We integrate CoSafe with seven state-of-the-art ADSes, and our evaluation on the InterDrive benchmark demonstrates that CoSafe improves the driving score of AVs by 70.03% on average, reduces safety violations by 87.80%, and accurately identifies unsafe AVs with an F3 score of 0.930, while satisfying real-time constraints.

### 论文解读
#### 摘要翻译
COSAFE面向混合交通中的异构自动驾驶系统，提供一个协同运行时安全层。它利用路侧单元汇集的车辆与基础设施信息识别场景风险，并在必要时协调多车轨迹，避免逐车安全修正造成新的相互冲突。
#### 方法动机分析
各车辆的规划器通常独立决策，单车看似安全的让行或制动，组合后可能形成碰撞、死锁或滞留。已有运行时保障多采用单车监控，难以处理车辆之间诱发的风险。COSAFE的假设是：RSU能从V2X接口取得车辆状态与意图，且车辆可以执行RSU返回的参考轨迹；在此基础上，一个不依赖具体ADS类型的场景级控制层可只覆盖不安全车辆。
#### 方法设计详解
输入包括RSU和车载BEV、车辆状态、ADS计划轨迹、地图及基础设施状态。首先将车载特征变换到RSU坐标系并融合，再预测道路参与者未来运动，匹配V2X车辆，构成统一场景。四类oracle检查碰撞、停车信号、死锁和滞留，并用冷却集合减轻预测抖动。系统为被标记车辆生成有限候选轨迹，通过分支定界搜索联合且相容的轨迹组合，赋予让行优先级；没有可行方案时保守停车。控制开关仅对危险车辆切换到安全参考轨迹，其余仍按原ADS运行。运行效率依赖RSU侧BEV融合、预测质量和候选搜索规模。
#### 方法对比分析
COSAFE不替代自动驾驶策略，也不要求各厂商统一内部模型；它在ADS外部增加干预接口。相比逐车接管，它明确估计多车互动，并把安全轨迹作为联合决策变量。相比集中式全车规划，它只修改判为不安全的车辆，试图保留原有系统行为。适合有路侧设施、V2X协作且车辆可接受参考轨迹的混合车流；对于无基础设施车辆或无法控制的参与者，只能通过感知间接考虑。
#### 实验分析（精简版）
在InterDrive的46条路线、7种ADS上，作者报告平均驾驶分提高70.03%、安全违规减少87.80%，联合干预活动周期约79.72ms。TCP消融中逐车贪心选择使碰撞数由21增至82，说明联合协调很关键。所有证据来自CARLA模拟，收益因ADS而异，残余死锁/滞留仍存在。
#### 实用指南
论文报告代码与数据开放，实验使用4张RTX 3090和128GB主机内存。迁移时需先统一BEV坐标、轨迹和V2X数据格式，校准oracle并验证最坏时延。作者覆盖四种风险并假设RSU处方式协作；真实道路上的通信丢包、感知误差、法规约束和故障处理尚需验证。
#### 总结
核心思想：场景级协同兜底
1. 融合RSU与车辆感知，预测交通参与者未来状态。
2. 用四类风险判断找出需干预车辆。
3. 联合搜索安全轨迹，仅覆盖危险车辆并回传执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.06204v1)
- [arXiv](https://arxiv.org/abs/2611.06204v1)

---

<a id='2611.05793v1'></a>
## [ZipDrift: Generating Multimodal Short-Horizon MPC Costs with One-Step Drifting Models](https://arxiv.org/abs/2611.05793v1)

**Authors:** Ke Zhang

**Published:** 2026-11-04

**Categories:** cs.RO, cs.LG, eess.SY

**Abstract:**

Short-horizon model predictive control (MPC) is fast but myopic. ZipMPC compensates by learning a context-dependent correction of the short-horizon cost that imitates a long-horizon expert. We observe that the optimal correction is often multimodal: when a slower opponent can be passed on either side, good corrections of opposite sign coexist, and the expert's choice between them hinges on tiny cost differences. Deterministic regression then averages modes or inherits inconsistent expert tie-breaking. We propose ZipDrift, which trains a one-step drifting generator over MPC cost corrections, tilted toward low closed-loop cost and learned only from cost evaluations of candidate corrections, without differentiating through the optimiser. A few generated candidates are ranked by a learned critic, and the model-based MPC then enforces dynamics and constraints. In a kinematic overtaking benchmark derived from ZipMPC, 53% of contexts are bimodal and the manual short-horizon MPC crashes in 28.5% of test episodes. With an eight-dimensional stagewise correction, ZipDrift is collision-free in all 600 test episodes and matches the best evaluated candidate, whereas an idealised ZipMPC-style regressor crashes in 4.0% (16.0% with inconsistent expert labels) and critic-ranked random candidates with the same budget in 2.0%.

### 论文解读
#### 摘要翻译
ZipDrift让短视MPC通过采样候选代价修正，获得更长时域专家所提供的决策偏好。针对超车决策中同时存在左右两种合理选择，论文用生成模型保留多模态，再由critic挑选低成本候选，最后仍由受约束的MPC执行动作。 该框架不让神经生成器直接控制车辆，学习组件只改变优化器代价，因此生成错误仍受MPC约束限制。

#### 方法动机分析
ZipMPC原先学习一个修正量回归长时域专家，但同一上下文可能存在多个相反方向的好修正，单点回归会将其平均或依赖不一致的专家选择。ZipDrift假设可以从候选修正的闭环成本评估中学习分布，并以critic筛选，比模仿唯一标签更稳健。
#### 方法设计详解
输入是当前控制上下文和短时域MPC的候选代价修正；模型训练一步漂移生成器，直接从成本评估学习修正分布，不需穿过优化器反向传播。生成时采样少量候选，可表示不同策略模式。critic基于上下文给每个候选估计闭环代价并排序，选中的修正再用于原MPC目标。于是学习模型扩展搜索偏好，MPC继续落实动力学、控制和约束。实验比较不同修正维度，并考察生成、critic与随机候选等因素。
#### 方法对比分析
和确定性回归相比，ZipDrift保留多峰决策；和端到端动作生成相比，它输出代价修正而非直接控制，因此仍由MPC约束执行。它适用于已有快速MPC但短视、且可离线评估候选成本的控制问题；若成本评估偏离实际目标，生成和排序会一同失准。
#### 实验分析（精简版）
在直线运动学超车基准中，53%上下文是双峰；8维修正在600次测试episode无碰撞，对比手工短视MPC的28.5%碰撞率、确定性回归基线的4.0%。该结果限于仿真和单个对手。 模型在测试时只对少量候选采样并用学习到的critic排序，比较预算需与随机候选保持一致；本结果反映候选质量与排序共同作用。 结果使用闭环碰撞作为安全指标，同时与同候选预算的随机排序方法比较；测试结果不提供真实道路暴露量或置信区间。部署时仍需定义可接受的碰撞概率、舒适度和控制平滑度指标，并检查采样候选是否覆盖不同超车方向。

#### 实用指南
论文以ZipMPC基准构造候选成本数据，需按相同动力学、超车设置和闭环评估协议复现；工程上应保留MPC约束作为最后执行层。迁移机器人需更新动力学和安全代价，并验证候选覆盖率及critic校准。 推理时应同步记录生成候选数、critic打分和MPC约束拒绝比例，避免只汇报碰撞率而忽略控制代价。

#### 总结
核心思想：生成多解，再由MPC兜底
1. 离线评估候选修正的闭环成本。
2. 生成多个可能的MPC代价校正并用critic排序。
3. 交给受约束MPC形成控制动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.05793v1)
- [arXiv](https://arxiv.org/abs/2611.05793v1)

---

<a id='2611.05801v1'></a>
## [WayPOP: A Panoramic Open-Set Panoptic Tracking Benchmark](https://arxiv.org/abs/2611.05801v1)

**Authors:** Rohit Mohan, Maximilian Luz, Swarnava Chowdhury, Yao Lu, Florian Drews, Thomas Nürnberg, Yakov Miron, Federico Tombari, Stefano Gasperini, Abhinav Valada

**Published:** 2026-11-04

**Categories:** cs.CV, cs.LG, cs.RO

**Abstract:**

Holistic perception for autonomous agents requires reliably segmenting and associating scene elements across time and camera views, including objects outside a fixed semantic taxonomy. Existing open-set panoptic benchmarks focus on individual images or views, while panoramic tracking benchmarks assume closed-set taxonomies. Consequently, these benchmarks do not evaluate whether unknown objects can be consistently segmented and tracked across synchronized multi-camera sequences. We introduce WayPOP, a panoramic open-set panoptic tracking benchmark built on real Waymo PVPS sequences augmented with diverse out-of-taxonomy 3D assets from Objaverse. Inserted objects are placed at fixed locations in the 3D scene and rendered using calibrated vehicle-mounted cameras as the ego vehicle and surrounding traffic move, providing temporally and cross-view consistent panoptic annotations and auxiliary LiDAR range observations. WayPOP contains approximately 670 curated assets instantiated as 980 unique unknown-object tracks, yielding 53,452 unknown-object mask annotations across 21,300 camera images. Our evaluation protocol defines a hierarchy of tasks from view-independent open-set panoptic segmentation to full panoramic open-set panoptic tracking, using known/unknown PQ and (w)STQ metrics. We benchmark representative open-set and open-vocabulary segmentation methods with temporal and cross-camera association. Our results reveal substantial limitations of current methods in truly open-set scenarios, highlighting robust unknown-object localization and identity association across time and views as major open challenges.

### 论文解读
#### 摘要翻译
WayPOP提出全景开放集全景分割跟踪任务与基准，关注视频中训练类别之外的物体：既要进行像素分割，也要在全景视角下跨时间保持对象身份。作者构建可控未知物体数据，并评估检测、分割和跟踪模块的组合表现。
#### 方法动机分析
机器人和自动驾驶环境总会遇到类别表之外的事物，但传统视频全景分割跟踪主要考察固定语义类别。仅看单帧开放集检测无法知道身份是否跨帧一致；传统总体指标也可能由已知类别主导。WayPOP的核心假设是，开放集能力必须在统一全景视频基准中按未知区域质量和时空关联分别测量。
#### 方法设计详解
数据集将3D未知资产插入全景视频背景，提供同步全景及前向画面、对象轨迹和像素掩码；规模包括670种资产、980条轨迹、53,452张掩码和21,300张图像。评估链路先通过开放集检测/分割产生未知对象候选，再由视频分割跟踪组件跨帧传播区域和身份。论文比较多个基线以及单视角、全景信息条件。wAQ刻画未知区域分割质量，wSTQ进一步纳入时间一致性，已知与未知类别分别报告，便于看清检测和跟踪短板。全景投影也带来视野边缘变形和跨视角关联问题。 训练和评测分别涉及开放集区域提取、像素级掩码与跨帧轨迹匹配；读者复现时需按论文协议保持相机投影和对象身份一致。 流程输入为同步全景/前向帧和开放集对象候选，输出逐帧未知对象掩码与持续轨迹ID；开源基线按统一协议组合检测、分割和时间关联。

#### 方法对比分析
相对闭集VPS数据集，WayPOP专门提供未知对象轨迹并显式度量开放集时空质量；相对只评估开放集检测的基准，它要求持续分割和身份关联。它适合比较开放词汇/未知类模块与跟踪器组合，但合成插入的资产分布可能与自然环境的新物体不同。 这项工作主要贡献是基准、数据及测量协议，基线组件本身并非全新跟踪器；因此其价值首先在于暴露未知对象时空关联差距。

#### 实验分析（精简版）
PoDS+DEVA的未知对象wSTQ评分为37.1分；即使以oracle掩码输入，未知对象wAQ评分也只有61.6分，提示身份关联等环节仍构成瓶颈。主要证据基于合成未知资产与全景视频构造，需谨慎外推到真实动态物体。 指标结果比较不同开放集模块与视频跟踪器组合，oracle掩码实验用来隔离分割误差；即便分割输入理想，剩余时空分数仍显示身份持续性难题。 PoDS+DEVA的未知对象wSTQ达到37.1分；oracle掩码下未知对象wAQ仍为61.6分，表明误差不止来自候选检测。

#### 实用指南
论文释放数据及基线实现。复现应保持全景投影、未知类别协议和时空身份标注一致，并分开报告未知/已知指标。迁移机器人或车辆前，应增加自然动态未知物体、传感器噪声、运动模糊及遮挡测试。 数据含同步视角和标注，推理链路应分别报告检测召回、区域质量和身份切换，而不要只用一个总体分数。

#### 总结
核心思想：全景未知物体时空评测
1. 在全景视频中加入未知对象及连续标注。
2. 组合开放集分割与视频身份跟踪。
3. 用区域和时空指标分别暴露误差。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.05801v1)
- [arXiv](https://arxiv.org/abs/2611.05801v1)

---

<a id='2611.05847v1'></a>
## [Time-to-Goal: Learned Temporal Metrics for Long-Horizon Planning with JEPA World Models](https://arxiv.org/abs/2611.05847v1)

**Authors:** Ruilong Yu, Fei Ye

**Published:** 2026-11-04

**Categories:** cs.RO, cs.CV

**Abstract:**

Joint-Embedding Predictive Architectures (JEPAs) support planning by comparing predicted latent states with an encoded goal. For this comparison to guide temporal progress, states closer in time to the goal should receive lower costs. However, training representations for prediction and distribution regularization does not directly enforce this ordering. We examine this mismatch in flat LeWorldModel (LeWM) by analyzing how its Euclidean goal cost changes with temporal separation. Our analysis identifies conditions under which states at different long offsets receive similar average costs, weakening their temporal distinction. To learn the missing ordering explicitly, we introduce Time-to-Goal (T2G), a lightweight metric trained on latent triples from offline trajectories. A margin ranking loss encourages smaller distances for temporally nearer states, requiring no rewards or policy labels. The learned metric then replaces the Euclidean planning cost while the JEPA encoder and predictor remain frozen. Diagnostics on AntMaze and FourRoom show that T2G preserves stronger temporal contrast and improves near--far ranking at long offsets. Across four navigation and manipulation benchmarks, T2G improves the average planning success rate from 32.7\% (flat LeWM) to 63.0\% and approaches or even surpasses the performance of the hierarchical H-JEPA baselines. These results show that learning temporal ordering for the goal cost can substantially improve latent planning even with a flat world model.

### 论文解读
#### 摘要翻译
Time-to-Goal提出一种从离线世界模型潜状态学习的时间距离度量，目标是评估当前状态距离目标还有多远。该度量可接在已有JEPA式潜空间规划器之后，不需要奖励或策略标签，帮助规划器选择更有希望到达目标的预测状态。
#### 方法动机分析
潜空间世界模型能生成未来状态，却缺少可靠的目标接近程度判断。像素差或普通嵌入相似度未必等于任务进度；奖励函数和专家策略标签又可能昂贵。作者假设同一离线轨迹中状态的时间先后可提供弱监督：越接近目标的后续状态应获得更短的预测时间距离。
#### 方法设计详解
先用冻结的LeWM/JEPA编码观察得到潜向量，再从离线轨迹构造锚点、近未来和远未来状态三元组。轻量度量头通过margin ranking loss学习相对时间排序，让较早状态到目标的距离大于后续状态。它新增约0.182M参数，不改变表示器，不需在线采样、环境回报或策略动作标签。规划器生成候选latent状态后，以该时间距离评估目标进展，并采用末端n步策略汇总规划结果，降低单个预测步造成的偶然性。设计重点是相对次序而非拟合精确秒数，因此可插接已有潜空间规划系统。 训练度量头时底层编码器冻结，使用离线状态三元组和排序损失；部署时规划器对每个生成未来状态调用度量头，而不是重新训练策略。

#### 方法对比分析
相较奖励学习，时间排序监督可以直接由离线轨迹获得；相较修改JEPA世界模型，它保持底层表示冻结、训练开销较小。适合已有可预测latent但目标距离估计不足的规划器。若轨迹时间顺序与成功进展不一致、目标状态定义模糊或预测latent漂移，学出的距离仍可能误导搜索。 与每步环境奖励不同，该模块学习的是轨迹时间次序；它补充世界模型规划，而不是取代状态预测或低层控制。训练时使用离线轨迹构造三元组，规划时以冻结表示器产生latent并重排候选。 相较依赖奖励标签的价值学习，它利用无标签轨迹中的时间顺序；相较改写世界模型，它只添加轻量指标。论文创新是面向目标的时间排序信号，不是新生成模型。

#### 实验分析（精简版）
四个模拟任务的平均last-n成功率由32.7%升至63.0%；AntMaze从18.0%升至58.7%，Push-T从40.0%到51.3%。评估使用冻结世界模型和三随机种子，尚无实机实验。 提升跨四任务不均，尤其迷宫任务更明显；Push-T仅增加11.3个百分点。三随机种子仍不足覆盖不同任务分布，论文也未报告物理机器人验证。

#### 实用指南
度量模块约0.182M参数，训练依赖同一任务轨迹中的时间三元组。实现时需确保目标条件、latent编码和时间采样匹配，并分任务报告成功率。迁移真实机器人前应测试观测遮挡、动力学变化、目标状态混淆与长程模型误差。 复现需按任务划分轨迹并避免相邻片段泄漏，训练三元组的时间间隔会改变排序难度；底层LeWM/JEPA保持冻结，新增头约0.182M参数。 配置重点是三元组的时间间隔、任务目标状态定义和候选规划步数，需在离线训练与在线规划时保持一致。训练只新增约0.182M参数的度量模块；推理依赖冻结的潜空间表示。

#### 总结
核心思想：以轨迹时间学习目标距离
1. 冻结世界模型编码离线观测。
2. 以时间先后构造状态三元组并训练排序度量。
3. 用目标距离重排规划器的未来候选。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.05847v1)
- [arXiv](https://arxiv.org/abs/2611.05847v1)

---

<a id='2611.06007v1'></a>
## [MotoVerse: A Benchmark for Predictive Perception of Motorized Two-Wheelers in Dense Unstructured Traffic](https://arxiv.org/abs/2611.06007v1)

**Authors:** Varun A. Paturkar, Shankar Gangisetty, C. V. Jawahar

**Published:** 2026-11-04

**Categories:** cs.CV

**Abstract:**

Motorized Two-Wheelers (MTWs) dominate personal transport across much of the Global South, yet remain largely overlooked in predictive perception research for autonomous driving and ADAS. We introduce MotoVerse, a benchmark for predictive perception from an MTW's ego-centric perspective. MotoVerse comprises 949 maneuver clips collected in dense, unstructured traffic and trajectory annotations for over 31,000 surrounding road agents, including over 16,000 MTWs, enabling two complementary tasks: Ego-Rider Maneuver Anticipation (ERMA) and Traffic Agent Trajectory Prediction (TATP). Each clip is further annotated with an MTW-specific challenge taxonomy comprising dynamic ego-motion, weak traffic structure and contextual occlusion, enabling challenge-aware evaluation. We benchmark established conventional prediction models, general-purpose Vision-Language Models (VLMs), driving-specific VLMs and vision-language-action models. Our experiments reveal three key insights: (i) video backbones reach most of their full-observation accuracy from the first fifth of a maneuver, whereas zero-shot VLMs need most of the maneuver to be observed; (ii) zero-shot VLMs, whether general-purpose or driving-specific, largely remain close to simple baselines on both tasks; a preliminary LoRA study shows that adaptation lifts anticipation sharply but leaves trajectory prediction behind conventional models; and (iii) the rider's own motion drives forecasting difficulty: for every model, including a static baseline, errors are highest during left turns and rise under dynamic ego-motion. We believe MotoVerse will serve as a challenging benchmark for safer and more robust predictive AI for intelligent two-wheelers. The dataset and evaluation code are publicly available at https://github.com/VarunIIITH/MotoVerse.

### 论文解读
#### 摘要翻译
MotoVerse面向密集无序交通中的摩托车等两轮机动车，提供预测性感知数据集与基准。它包含轨迹预测以及早期机动/风险预测任务，让研究者评估系统能否在动作真正发生前，从有限视频历史判断两轮交通参与者的未来行为。
#### 方法动机分析
两轮机动车体积小、机动自由，常穿行车辆间隙，和汽车主导的数据分布不同。仅做当前检测或已发生轨迹拟合，无法回答系统是否足够早地预判转向、超车等行为。MotoVerse的假设是通过真实道路片段、交互窗口和合成补充，可更针对性地衡量预测提前量与预测误差。
#### 方法设计详解
基准包含949段视频、超过31k条轨迹和超过16k个交互时间窗，来自印度真实道路采集与合成/仿真数据。真实部分约50小时、700公里，使用单台1080p 30FPS GoPro和16名骑者，划分566/113/270段用于训练、验证和测试。TATP根据历史轨迹预测未来位置；ERMA则评估视频观察到不同阶段时能否判断机动或风险。作者以SGNet等模型建立轨迹基线，并比较静态预测、不同历史可见比例及预训练/LoRA微调，以刻画误差与提前量的权衡。 TATP把观察历史映射为未来位置序列，按轨迹误差衡量；ERMA改变观察窗口长度，要求模型在运动尚未完全显现时判断机动。两项任务共同检验预测内容与预测时机。训练/测试按566、113、270段划分。 真实与模拟视频经过统一主体轨迹标注，模型接收观察窗内的历史运动/外观线索并输出未来轨迹或机动类别；测试通过截短历史窗模拟更早预测。

#### 方法对比分析
相较汽车中心轨迹数据，MotoVerse强化两轮主体及交互；相较仅报告最终轨迹误差，它增加早期识别设定。可用于预测感知和驾驶规划前端研究，但数据来源地域、视角和参与者有限，模型排序可能随采集城市改变。 相较仅用常规车辆轨迹预测，本文强调两轮参与者；相较只判断已发生动作，它显式改变视频可见比例测早期识别能力。贡献主要是领域数据与评估设置，而不是一个通用驾驶控制器。

#### 实验分析（精简版）
SGNet的常规ADE为51.8，静态预测基线为94.3。ERMA仅观察20%视频时准确率约50%，较高结果依赖更长上下文和后续LoRA微调；人类基准约79%。基准评估不包含闭环避险效果。 ADE受历史窗口与预测长度影响，ERMA准确率也随可见比例提高；两项指标不可直接互换。模型预测只作为离线评测，没有闭环碰撞率或规划收益。 SGNet相对静态基线的ADE从94.3降至51.8；这是轨迹预测误差而非避碰成功率，ERMA的提前量结果另以分类准确率报告。

#### 实用指南
论文给出数据划分和基线，可从轨迹预测与提前量两部分复现。评估必须明确可见历史比例、预测时距和数据源，避免将仿真收益混同真实数据表现。迁移自动驾驶需增加跨城市、天气、夜间和多传感器验证。 训练细节与划分按数据集基准配置复现；部署决策时应将置信度与动作风险结合，而不是把分类概率直接当控制信号。

#### 总结
核心思想：提前预测两轮交通行为
1. 汇集两轮参与者道路视频与轨迹。
2. 分别评估未来轨迹和早期机动预测。
3. 用历史比例与基线比较预测提前量。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.06007v1)
- [arXiv](https://arxiv.org/abs/2611.06007v1)

---

<a id='2611.06020v1'></a>
## [METRIC-GS: Selective Metric-Scale Estimation for 3D Gaussian Splatting](https://arxiv.org/abs/2611.06020v1)

**Authors:** Sooyeun Yang, Jongseong Brad Choi

**Published:** 2026-11-04

**Categories:** cs.CV

**Abstract:**

A 3D Gaussian Splatting (3DGS) reconstruction inherits arbitrary units from structure-from-motion, so it cannot be measured or used to place AR content until someone supplies a scale. Both standard recipes are fragile on real captures: on five indoor scenes with tape-measured ceilings, a quantile floor-to-ceiling heuristic errs by 27% and metric-depth alignment by 35-71% across three networks. We present METRIC-GS, a selective scale estimator for real indoor photographs: it reads the scale off the reconstruction's own semantic content and, just as importantly, refuses when the evidence is thin. A frozen 2D teacher labels Gaussians by visibility-aware voting; instance-level size priors for architectural and furniture classes (doors, tables, the floor-to-ceiling gap) are fused by a robust maximum-likelihood step; and an inlier-mass test turns thin evidence into an explicit abstention that doubles as a router to a heavyweight metric model. On 47 ScanNet++ scenes with laser ground truth, with a configuration frozen before public evaluation, the estimator alone cuts the heuristic's mean error from 21.1% to 16.6% (9.3% on the 28 scenes it acts on) at ~11 s per scene. MASt3R remains the more accurate standalone estimator (9.9%), but routing only our refusals to it matches running it everywhere with 60% fewer invocations; this selective hybrid is the configuration we advocate. Visibility-aware voting raises the scenes the estimator can act on from 16 to 28 at unchanged deployed accuracy. Applied unchanged to 13 phone and drone captures of our own, on which MASt3R fails its own registration gate 12 times, the system improves all five captures it acts on and degrades none. We also report that a current feed-forward metric model scores 2.3% on a benchmark inside its training mixture and 28-35% on captures it has never seen: a published metric-scale number is a best case, not a deployment estimate.

### 论文解读
#### 摘要翻译
METRIC-GS针对3D Gaussian Splatting重建缺少绝对尺度的问题，提出一种选择性尺度估计方法。它利用场景中已识别对象的典型尺寸和多视角投影估计尺度，并根据置信度决定直接输出、弃答或调用更昂贵的几何匹配模型。
#### 方法动机分析
仅凭稀疏照片重建出的3D场景可能在整体缩放后仍保持相似投影，因此不能直接用于真实测量或导航。逐个场景都使用重型匹配模型成本高；而语义对象的尺寸先验可能提供廉价线索。作者假设，多物体、多视角的可见投影证据经稳健聚合可给出尺度，并以置信度识别不可靠场景。
#### 方法设计详解
输入为3DGS高斯场景及多视角图像。系统检测语义对象，结合对象尺寸先验和高斯投影，估计候选全局尺度。visibility-aware投票降低遮挡、截断和低质量视图的权重，再用稳健最大似然估计聚合尺度并计算不确定性。若置信度达到阈值就直接预测；否则选择拒绝输出，或路由至MASt3R等高成本估计器。因而推理流程在覆盖率、精度和算力间可调，重模型不必用于所有场景。 每个对象提供一个尺度假设，系统先经可见性过滤，再进行稳健聚合；置信门控决定是否调用MASt3R，体现覆盖率—精度—算力的折中。估计器冻结测试便于区分已训练适配与零样本表现。 输入到输出包括对象检测、可见投影计票、全局稳健拟合和阈值式选择：高置信度直接返回尺度，低置信度切换MASt3R或拒答。

#### 方法对比分析
与一律调用几何匹配相比，METRIC-GS把语义尺寸先验作为便宜筛选器，置信不足时才升级；与无条件全场景输出相比，允许弃答以控制错误。关键新意是可见性意识的多视角尺度投票和按置信度路由。适合室内照片3DGS场景；若物体尺寸异常、部分物体被遮挡或训练先验偏差，尺度会误导。
#### 实验分析（精简版）
47个ScanNet++激光真值场景中，冻结估计器平均误差16.6%，主动子集9.3%；路由后误差9.4%，MASt3R调用减少60%。只覆盖室内照片，部分可见对象仍可能产生高置信错误。 主动集只含接受直接尺度预测的场景，选择性机制牺牲部分覆盖率来降低平均误差，因此部署时要同时报告覆盖率、错误率和路由成本。 冻结估计器全场景误差16.6%，只看接受估计的28个场景为9.3%；路由后9.4%且重模型调用减少60%，需同时理解误差和覆盖率。

#### 实用指南
复现需重建同一3DGS坐标尺度，匹配对象类别与尺寸先验，并校准弃答/路由阈值。论文评估含MoGe-2数据污染讨论。迁移到机器人地图可减少常态尺度估计成本，但动态目标、室外尺寸分布和传感器变化需另行验证。 ScanNet++激光测量提供度量真值，复现时需核对场景划分和相机内外参，并在目标应用重新估计各类别真实尺寸先验。

#### 总结
核心思想：不确定时升级尺度估计
1. 从多视角识别对象并取得投影尺寸线索。
2. 按可见度投票、稳健估计场景尺度与置信度。
3. 低置信时弃答或路由到高成本模型。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.06020v1)
- [arXiv](https://arxiv.org/abs/2611.06020v1)

---

<a id='2611.06089v1'></a>
## [STARE: Scene Text-Aware multimodal Retrieval for Everyday objects](https://arxiv.org/abs/2611.06089v1)

**Authors:** Kento Tokura, Ryosuke Korekata, Daichi Yashima, Kei Katsumata, Kanon Amemiya, Miyu Goko, Komei Sugiura

**Published:** 2026-11-04

**Categories:** cs.CV

**Abstract:**

In this study, we focus on service robots that retrieve and transport everyday objects based on free-form instructions that may refer to visual features and scene text. This is challenging because it requires a multimodal understanding of the visual features and spatial relationships of objects, as well as scene text embedded in the environment. To address this, we propose Scene Text-Aware multimodal Retrieval for Everyday objects (STARE), which leverages visual and textual cues in response to free-form instructions. STARE introduces Crosslingual Visual Prompt to enhance scene text interpretation, and models the complex relationships between named entities in the instructions and their corresponding objects. We evaluate STARE using two novel benchmarks, GoGetIt and TextCaps-test, both of which consist of real-world indoor/outdoor images and instructions involving object manipulation and navigation. The results show that STARE outperforms all baseline methods on standard multimodal retrieval metrics. Furthermore, in physical experiments conducted under a zero-shot transfer setting, STARE achieves an overall success rate of 80%. Our project page and code are provided in the Supplementary Materials.

### 论文解读
#### 摘要翻译
STARE研究机器人日常物品检索：当用户通过品牌、标签或文字描述寻找对象时，系统同时利用自然语言、图像外观和场景文字。方法由查询扩展、场景文字感知视觉表示和实体词汇重排组成，追求准确且较快的检索。
#### 方法动机分析
普通视觉检索可能忽略包装和环境中承载的文字，纯语言模型分析则耗时；开放式描述中的实体名称又常需精确匹配。作者认为OCR文本和视觉外观互补，若在表示学习与重排两处都使用文字，可兼顾语义泛化和精确实体识别。 挑战不仅是视觉相似对象难区分，也包括查询实体在包装上可读、却未进入图像嵌入的情况；系统希望把这类文字证据直接纳入召回和精排。

#### 方法设计详解
查询先进入MFIE模块，抽取并扩展实体、属性等线索。场景端通过OCR获取候选文字区域，STVE将场景文本与视觉区域编码为场景文字感知的多模态表示，使查询与对象在同一检索空间比较。初始召回后，STRR识别查询中的命名实体并执行词汇级重排，提升名称精确匹配。该流程从查询到候选召回再到实体纠偏，融合软语义相似和显式词面证据。作者使用多组检索数据及机器人硬件试验评估，并报告训练配置为20轮、AdamW学习率1e-4、batch 128。
#### 方法对比分析
相比只依赖视觉嵌入的方法，STARE显式利用可读文字；相比每次调用大型生成模型，它把任务拆成可缓存的检索与重排模块。它适用于物品标签、包装、菜单或货架文字清晰可读的场景；OCR漏检、文字遮挡和用户别名会削弱效果。
#### 实验分析（精简版）
Recall@10在GoGetIt RefText、指令、TextCaps、LTRRIE分别为94.4、92.8、95.4、86.6。硬件100次试验成功率80%、平均2.0秒，对比GPT-4o的21.3秒。试验规模有限，失败率仍不可忽略。 检索Recall@10衡量候选召回，而硬件成功率还受到定位、导航和抓取影响；80%意味着仍有失败，速度比较也需在各自硬件与推理设置下理解。 不同公开检索集上Recall@10达到94.4、92.8、95.4和86.6；硬件取物100试验80%成功、2.0秒，对照模型21.3秒。两种评估口径不同。

#### 实用指南
复现需要对齐OCR、实体提取、跨模态特征和词面排序规则，并报告检索精度与实际抓取成功率。部署仓储机器人需评估字体、光照、语言与相似标签，并设计置信度阈值和失败恢复。 训练批量为128、共20轮，硬件实验100次；复现务必注明OCR模型、相机视角与端到端成功定义。

#### 总结
核心思想：让文字参与视觉检索
1. 扩展查询并提取关键实体。
2. 联合编码OCR场景文字与视觉对象。
3. 召回后用实体词面信息重新排序。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.06089v1)
- [arXiv](https://arxiv.org/abs/2611.06089v1)

---

<a id='2611.06184v1'></a>
## [Can Astra Solve RoboMME without Breaking the Bank?](https://arxiv.org/abs/2611.06184v1)

**Authors:** Bingao Chen, Haoquan Fang, C. Karen Liu

**Published:** 2026-11-04

**Categories:** cs.RO

**Abstract:**

Robotic manipulation often requires remembering information that is no longer visible, such as an object's location before occlusion or the order of a demonstrated action sequence. A large vision-language planner can read this history, but calling it before every action chunk is slow and costly. We present a three-tier system for the RoboMME benchmark. GPT-6 Astra (System 2) selects a grounded subtask from stored visual history, a $π_{0.5}$ vision-language-action policy (System 1) executes it, and a fine-tuned Qwen3-VL-4B monitor (System 1.5) checks after each action chunk whether the subtask is finished. The instruction is kept until the monitor reports completion or a task-specific event occurs, while the policy keeps acting on new observations. On RoboMME, the system succeeds in 79.13% of episodes with an average of 3.63 Astra calls and 80.84 seconds per episode. This is 34.62 points above the best published non-oracle baseline and 4.95 points below GroundSG+Oracle, which gives the low-level policy ground-truth subgoals.

### 论文解读
#### 摘要翻译
论文研究Astra分层规划系统能否以较少高层模型调用解决RoboMME机器人任务。系统把高层视觉语言规划、低层动作块策略和事件监控结合起来：低层连续执行，遇到关键事件再要求高层重新规划。
#### 方法动机分析
长任务若每一步都调用大型规划器，成本和等待时间会很高；若一直执行固定动作块，又可能无法及时处理偏差。作者假设多数时刻可以由低层策略处理，只有目标完成、失败或异常等事件需要昂贵的高层推理，从而同时保持成功率并减少调用。
#### 方法设计详解
系统分三层：Astra高层规划器按任务目标生成子目标；π0.5低层策略每次输出16步动作块；Qwen3-VL-4B监控器读取视觉状态和执行反馈，判断是否继续、终止当前子目标或触发重规划。事件触发机制减少固定频率的高层规划调用。整套系统在RoboMME任务上运行，平均规划器调用次数、时间和成功率共同衡量效果。任务配置仍需人工提供，仿真器在规划阶段暂停，因此测试执行节奏与真实连续机器人不同。
#### 方法对比分析
相较每个动作都调用高层模型，事件触发方案把高层推理集中到状态转折点；相较纯动作块执行，视觉监控能为失败恢复提供入口。主要贡献是系统级组合和调用调度，而非新的底层动作模型。适用于任务结构清晰、可检测关键事件的操作基准。
#### 实验分析（精简版）
RoboMME总体成功率79.13%，最佳已发表非oracle基线44.51%；oracle为84.08%。平均Astra调用3.63次、任务时间80.84秒。StopCube和InsertPeg仍是明显失败项，未提供模块消融或监控错误率。 对比结果来自不同公开方案和统一任务测试，但无系统消融，无法独立归因于调度、监控器或底层策略。规划阶段暂停仿真时钟，故80.84秒是特定运行口径。 作者报告总体成功率79.13%，领先最佳非oracle基线44.51%，同时比oracle的84.08%低4.95个百分点；800个episode结果不能替代外部实机复现。

#### 实用指南
实现依赖Astra、π0.5和Qwen3-VL-4B组合，动作块长度16；复现时应记录任务级配置、调用次数、恢复轨迹和真实墙钟时间。迁移实机需验证异步执行、安全中断、监控误报漏报和物理延迟。 复现需要固定策略版本、动作块长度、监控触发规则和重试预算，并同时统计成功、调用次数、失败恢复与墙钟延迟。另需衡量手工配置任务目标的工作量，并在实体机器人上报告监控误报、漏报及动作中断后的安全状态。还应核查未见任务的手工配置负担，并记录监控误报、漏报和动作中断后的状态。

#### 总结
核心思想：关键事件时才高层重规划
1. 高层规划拆分任务子目标。
2. 低层策略连续执行动作块，视觉监控状态。
3. 遇到关键事件时触发重规划或恢复。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.06184v1)
- [arXiv](https://arxiv.org/abs/2611.06184v1)

---

<a id='2611.06544v1'></a>
## [MKD2World: A Typed Spatial MarkDown World Model For Web Agents](https://arxiv.org/abs/2611.06544v1)

**Authors:** Zhihao Nan, Sifan Zhou, Yiming Cheng, Jinkai Zheng, Zheng Zhu, Si Shi, Ming Li, Fei Yu

**Published:** 2026-11-04

**Categories:** cs.CV

**Abstract:**

Graphical user interface (GUI) agents can improve decision-making with world-model foresight, provided that predicted states accurately reflect post-action interfaces. Existing state representations fail to jointly capture interface layout, UI semantics, and explicit locations. We introduce Typed Spatial Markdown (TSM), a Markdown-style textual state representation for desktop web interfaces. TSM serializes block-level and atomic-level UI elements according to their spatial relationships, jointly encoding their types, visual semantics, explicit locations, optional control states, and hierarchical membership. To train a screenshot-to-TSM renderer, we construct AtomBlock-WebUI, a dataset with precise annotations at both granularities. By injecting element-type labels into HTML and directly extracting bounding boxes after rendering, we avoid annotation errors caused by post-hoc inference from webpage structures. Building on TSM, we propose MKD2World, a desktop web world model trained in two stages for current-state description and action-conditioned next-state prediction. During inference, the web agent queries MKD2World as needed and performs a ``digest'' step, in which it interprets each predicted state and extracts task-relevant evidence for selecting the best action. Experiments show that MKD2World-9B achieves an EleF1 score of 26.50 on the out-of-domain test set, outperforming GPT-5.6-Sol(14.98) by 11.52 points. Across Online-Mind2Web tasks, MKD2World-9B improves the success rate of the Qwen3.5-35B web agent from 22.94% to 37.61%, achieving competitive performance among leading open-source GUI world models. Code is available at https://github.com/OmniAI-Lab-CUHKSZ/MKD2World.

### 论文解读
#### 摘要翻译
论文提出Typed Spatial Markdown（TSM），将桌面网页界面压缩为带空间位置与层级的文本状态，并据此训练MKD2World预测用户操作后的页面状态。网页智能体可以把预测变化整理成证据摘要，再比较候选动作，降低试错式操作。
#### 方法动机分析
已有网页世界模型使用变化描述、DOM/无障碍树、HTML或生成图片，各自会遗漏布局位置、元素语义，或带来重建成本与小字失真。作者认为，智能体不必生成完整像素页面，但需要同时知道控件是什么、在哪、状态如何以及处于何种区块。TSM的假设是：这组结构化空间信息足够支持候选动作之间的状态比较。
#### 方法设计详解
TSM把界面组织为区块和原子元素，记录类型、可见语义、归一化边界框、可选控件状态和层级关系。AtomBlock-WebUI先给HTML注入类型标签，再渲染截图并直接从对应节点读取框，得到约9.7k页的双粒度标注。截图转TSM时，检测器预测类型和位置，OCR识别文本，小型视觉模型补充图标及图片语义，随后按包含关系整理UI树。MKD2World基于Qwen3.5-9B进行两阶段微调：阶段一学习截图到当前TSM，阶段二学习截图与候选动作到后续TSM。Base-Focus Element Loss强调新增和更新元素，margin损失让变化状态优先于当前状态。推理时，智能体提出候选动作，模型分别预测后状态；智能体抽取任务相关变化、检查动作意图是否得到支持，再选择执行。置信度高时可直接执行。 阶段二训练以动作类型均衡的转移为目标；推理的digest包含预测变化、意图一致性与任务推进证据，避免只凭模型分数选动作。

#### 方法对比分析
TSM比自由文本变化描述更完整，比DOM/AXTree更显式地给出位置，比HTML或像素生成更易直接比较且成本较低。贡献由标注流程、空间状态表示、转移预测损失和digest决策流程组成。它适合界面布局稳定且可见元素决定操作结果的桌面网页；若后台逻辑、动态内容或动画主导结果，静态状态描述就不够。
#### 实验分析（精简版）
域外预测集EleF1为26.50，高于GPT-5.6-Sol的14.98；Qwen3.5-35B在109项在线Mind2Web任务的成功率由22.94%升至37.61%。提升依赖智能体骨干，GUI-Owl的困难任务表现反而下降；在线测试规模与动态页面覆盖有限。 在线任务按完整轨迹由评审模型判定，删除网络不稳定任务后剩109项；该协议提供真实网页交互证据，但不等同广泛网站覆盖。 域外EleF1为26.50，对照GPT-5.6-Sol为14.98；在线Qwen3.5-35B成功率从22.94%到37.61%，评审器按完整操作轨迹判分。

#### 实用指南
AtomBlock-WebUI、模型代码公开。复现需统一网页渲染尺寸、类型词表、边界坐标和域隔离；论文阶段二约31k转移、底座9B并使用LoRA，训练在A800进行。迁移到桌面软件或机器人操作面板有潜力，但必须新增动态状态和实际交互验证。 训练设置使用Qwen3.5-9B做世界模型、LoRA微调，并以A800训练；部署要监视TSM解析成功率和digest证据是否忠实。

#### 总结
核心思想：用空间文本预演网页
1. 把网页截图转为有类型、位置和层级的TSM。
2. 根据截图与候选动作预测后续TSM。
3. 摘要对比预测证据，再选择是否执行操作。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.06544v1)
- [arXiv](https://arxiv.org/abs/2611.06544v1)

---

<a id='2611.05766v1'></a>
## [ME^2FDiff: Multiscale Expert Enhancing Feature Diffusion in Residual Network for Brain Tumor Segmentation](https://arxiv.org/abs/2611.05766v1)

**Authors:** Afrida Rahman, Bijoy Kundu

**Published:** 2026-11-04

**Categories:** cs.CV

**Abstract:**

Automated segmentation of post-treatment glioma from MRI is essential for accurate diagnosis and treatment planning, yet remains difficult. Surgery and chemoradiotherapy cause structural changes in brain like resection cavities, edema, and fibrosis, that alter imaging appearance and mimic tumor recurrence. Contemporary segmentation architectures attain high accuracy with two limitations. First, the bottleneck originates from the deepest encoder stage, so one representation must serve sub-regions that differ widely in size and appearance. Second, they produce the segmentation in a single forward pass with no mechanism for refinement. We propose ME^2FDiff, a single-model architecture, addressing both. An attention-conditioned spectral diffusion block refines the bottleneck. It uses the image-conditioned latent diffusion process with two processes: a spatial pre-attention gate and a spectral fusion block whose timestep-adaptive bandpass filter selects the informative frequency band at each noise level. A multiscale expert-enhanced feature block fuses the intermediate encoder scales into the bottleneck through expert-selected shared tokens, guaranteeing balanced utilization without an auxiliary loss. On the BraTS 2024 post-treatment glioma dataset, ME^2FDiff achieves the best results on every tumor sub-region among all single-model baselines (91.23% Dice, 84.52% IOU, 2.13 mm HD95 on whole-tumor) and surpasses a three-expert ensemble on both overlap metrics at roughly one-third of its inference cost. Against recent state-of-the-art methods retrained under identical conditions, it maintains the strongest performance with the fewest parameters of any competing model. These findings demonstrate that a single, compact network can absorb the specialization, usually sought through ensembling or larger architectures.

### 论文解读
#### 摘要翻译
ME2FDiff是一种面向后治疗脑胶质瘤MRI分割的单模型架构。它把紧凑掩码潜变量上的条件扩散精炼，与多尺度专家特征融合放进SegResNetVAE，在避免三模型集成开销的同时增强不同肿瘤区域的刻画能力。
#### 方法动机分析
后手术及放化疗改变脑组织外观，治疗效应可能与复发肿瘤相似，导致小区域和边界难分。普通编码器瓶颈主要来自最深尺度，浅层细节未能在瓶颈直接融合；单次分割也没有后续校正步骤。作者的假设是，压缩掩码后做条件扩散仍能改善边界，而将各尺度特征用专家路由融合可以在一个网络内获得类似集成的互补性。
#### 方法设计详解
输入为T1n、T1c、T2w和FLAIR四模态MRI，非零脑区各自归一化、裁剪和重采样到128³。SegResNetVAE编码出多尺度特征与瓶颈。ACSDiff先以空间预注意门控筛选有用通道和位置，再将图像语义与受噪掩码潜变量映射到频域交互；频率门控随扩散时间步改变，噪声较大时保留粗结构，后期逐渐强化细节。掩码压缩至5×16³潜变量，训练时学习噪声预测，推理时从高斯噪声条件生成掩码潜表征。随后ME2FB把不同编码尺度投到统一维度和瓶颈网格，由expert-choice分配token，每个专家处理固定比例以避免专家塌缩；聚合结果与瓶颈残差融合后解码。联合损失包含扩散噪声预测、Dice/Tversky/Focal分割目标和VAE正则，并提高小区域漏检代价。
#### 方法对比分析
与三模型融合相比，ME2FDiff把扩散精炼和尺度路由内置在单网络，不需要多个完整模型推理；与常规SegResNetVAE相比，它既改变瓶颈的多尺度信息来源，也为掩码增加迭代式潜空间修正。其优势面向体数据和有限计算预算；频域扩散增加实现复杂度，且任务外迁移需要重新处理MRI协议差异。
#### 实验分析（精简版）
BraTS2024后治疗glioma的163例测试集上，整肿瘤Dice 91.23%、IoU 84.52%、HD95 2.13mm；三模型MoE训练每轮39.87分钟，本文单模型10.92分钟。重切除腔仍是难点，且证据来自单一数据集切分。不同亚区的排序和边界距离未必同步改善；作者虽用重复种子分析增强肿瘤组件，但尚不能据此断言其他中心同样有效。 数据按70/20/10分成训练、验证、测试，最多150轮并早停；单次固定划分的结果仍需要外部中心和患者级独立复核。 对照SegResNetVAE，整肿瘤Dice从90.65%提升到91.23%；对比三模型MoE时平均Dice更高且训练更快，但部分边界度量并非全部获胜。

#### 实用指南
训练设定为最多150轮、早停40轮、每GPU batch size 1、四张A6000/A100；预处理和病例级划分需严格复现。代码未见于本文主文明确链接，应先确认可用实现。迁移到其他器官或MRI协议前需重做标注、归一化和外部验证，不能把基准准确率直接当临床性能。 训练每GPU batch size为1，使用四张A6000/A100及AdamW，峰值学习率1e-4；各模态顺序和固定重采样/裁剪必须一致。

#### 总结
核心思想：潜空间精炼，多尺度路由
1. 多模态MRI编码为瓶颈与多尺度特征。
2. 在压缩掩码潜空间条件扩散并融合各尺度。
3. 解码得到肿瘤亚区分割。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.05766v1)
- [arXiv](https://arxiv.org/abs/2611.05766v1)

---

