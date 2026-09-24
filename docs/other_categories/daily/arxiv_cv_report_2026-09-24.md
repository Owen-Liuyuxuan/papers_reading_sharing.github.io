time: 20260924

# Arxiv Computer Vision Papers - 2026-09-24

## Table of Contents

1. [DAVIO: Dense Monocular-Inertial SLAM with Feed-Forward Initialization and Pose-Conditioned Mapping](#2609.27702v1)
2. [Latent evolving World Action Model](#2609.27455v1)
3. [InternW0: A Foundational Physical World Model for Efficient Real-World Interactions](#2609.27656v1)
4. [Kairos: Grounded Forecasting of Presence and Directional Flow in 4D Scene Graphs](#2609.27467v1)
5. [CoPRE: Improving Sensitivity in Proprioceptive Contact Detection for Low-Cost Robot Arms](#2609.27381v1)
6. [BEE: Intervention-Adaptive Real-World Reinforcement Learning with Vision-Language-Action Models](#2609.27450v1)
7. [From LiDAR Maps to Visual Localization: Unified Visual Association for Robust Point-Line-Plane Pose Estimation](#2609.27363v1)
8. [Behavior-Aligned Action Tokenization for Robot Policy Learning](#2609.27513v1)
9. [Surgical Kinematics from Monocular Video with Learned Articulated Motion Constraints](#2609.27227v1)
10. [Reflection-Aware Reasoning for Non-Line-of-Sight Pedestrian Localization](#2609.27346v1)

---

## Papers

<a id='2609.27702v1'></a>
## [DAVIO: Dense Monocular-Inertial SLAM with Feed-Forward Initialization and Pose-Conditioned Mapping](https://arxiv.org/abs/2609.27702v1)

**Authors:** Jaafar Mahmoud, Arthur Movsesyan, Mikhail Iumanov, Sergey Kolyubin

**Published:** 2026-09-23

**Categories:** cs.RO, cs.CV

**Abstract:**

A camera and an IMU are the minimal sensor setup for metric localization and dense mapping, yet classical visual--inertial filters must wait for parallax before they start and then retain only sparse landmarks. Feed-forward geometry models, in contrast, predict dense structure from a few images but provide neither metric scale nor gravity. We present DAVIO, which uses a single multi-view depth model, Depth Anything~3, for both start-up and mapping. At start-up, a five-image window and preintegrated IMU measurements form a feature-free linear system. Its robust, conditioning-checked solution bootstraps a VIO filter through buffered replay. During tracking, the filter's metric poses condition the depth model. Residual scale is corrected only along viewing rays, which preserves the metric camera baselines, and a gravity-preserving submap graph with drift-gated revisits refines the map. On EuRoC, DAVIO starts markedly earlier, reduces the localization error, and maps more accurately than SOTA feed-forward mappers given identical poses. On building-scale ORI sequences, DAVIO is on bar or better than SOTA mappers on the same odometry, and degrades far less when GT poses are replaced by real odometry. We release the code of DAVIO, a real-time dense metric SLAM system, to the community.

### 论文解读
#### 摘要翻译
DAVIO 面向单目相机与 IMU 的实时稠密度量 SLAM。经典 VIO 要等待足够视差才能恢复尺度，且只维护稀疏路标；前馈几何模型虽能快速输出稠密结构，却缺少尺度和重力。DAVIO 让 Depth Anything 3（DA3）同时负责启动和建图，并用 IMU 把其几何锚定到度量坐标。
#### 方法动机分析
系统抓住两类传感器的互补性：DA3 在少量图像上立即给出稠密几何，惯性测量提供尺度、重力和运动物理量。核心假设是短窗口内 DA3 的相对几何足以与 IMU 预积分共同求解状态；当深度、相对旋转或运动条件不可靠时，系统宁可拒绝窗口。
#### 方法设计详解
启动阶段每隔 0.2 秒取五帧图像。DA3 输出深度、置信度、内参和相对位姿，每帧最多按 4×4 网格采样 100 个高置信度点。射线约束与 IMU 预积分组成 7 维线性系统，求尺度、初始速度和重力；100 个 LMedS 假设抑制动态物体及边界异常点，再用 Huber 损失细化，并以 softplus 保证尺度为正。通过内点率、残差、奇异值比、速度和重力方向门控后，状态经最多 8 秒缓冲回放交给 OpenVINS。跟踪时，VIO 的度量相机位姿输入 DA3，形成重叠子地图。残余尺度只沿像素视线修正，不改变 VIO 基线；共享图像、相邻子地图和经过漂移门控的回访约束共同优化地图位姿与深度尺度。
#### 方法对比分析
相较 OpenVINS，DAVIO 具备更早启动和稠密表示；相较 ScaRF-SLAM 等位姿条件建图器，它新增了 feature-free 点—惯性初始化、射线方向尺度修正及保持重力的回访图。与依赖特征跟踪的 ORB-SLAM3 相比，它把 DA3 的稠密预测用于启动，并将地图帧优化与 VIO 状态解耦；与 MASt3R-Fusion、VGGT-SLAM2 相比，重点是固定度量基线后只优化残余深度尺度。VIO 滤波状态持续独立运行，地图优化只修正输出地图帧，适合需要快速稠密地图的机器人。
#### 实验分析（精简版）
在 EuRoC 11 条序列上，DAVIO 首个状态中位时间为 2.43 s，原生 OpenVINS 为 3.65 s；Machine Hall 的质量惩罚时间由 4.68 s 降至 2.30 s。EuRoC MH 的 ATE 相对原生 VIO 降低 29–45%。在 ORI 五条建筑尺度序列上，平均误差由未条件化的 0.219 m 降至 0.080 m；Vicon Rooms 中位 F@10 为 0.58，而同位姿 ScaRF-SLAM 为 0.45。去掉回访会使 MH_03 ATE 从 0.101 m 增至 0.158 m。结果依赖 DA3 几何一致性及传感器标定。
#### 实用指南
作者提供代码链接 https://be2rlab.github.io/DAVIO/ 。复现需准备图像、IMU、标定和时间同步信息；DA3-Base 使用 504 像素输入，实验硬件为 RTX 3060 Laptop 6 GB。应保留五帧窗口、LMedS 门控、子地图的 5 cm 基线和 15 cm 拟合 RMSE 条件，以及八次 LM 更新。迁移到其他机器人时需重设坐标系、惯性噪声和时间偏移，并验证 DA3 在新场景中的一致性。
#### 总结
核心思想：惯性锚定前馈稠密几何
速记：
1. 五帧 DA3 几何与 IMU 解尺度重力。
2. 鲁棒门控后回放启动 VIO，并以度量位姿条件化 DA3。
3. 沿视线修正深度，以回访图融合子地图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27702v1)
- [arXiv](https://arxiv.org/abs/2609.27702v1)

---

<a id='2609.27455v1'></a>
## [Latent evolving World Action Model](https://arxiv.org/abs/2609.27455v1)

**Authors:** Xueji Fang, Boqiang Duan, Hua Wu, Jingdong Wang, Guo-Jun Qi

**Published:** 2026-09-23

**Categories:** cs.CV, cs.RO

**Abstract:**

World Action Models (WAMs) jointly model action generation and environment dynamics and are mostly built on pretrained Video Diffusion Models (VDMs). In VDM-based WAMs, observations are first encoded by a VAE, and the resulting compressed latents are then processed by large video diffusion backbones to extract effective features for action generation. However, this paradigm ties WAM performance and training cost to large-scale video generation pretraining, limiting WAM efficiency and scalability. In this paper, we theoretically and empirically investigate how visual representations affect action generation in WAMs. Our results show that predictive embeddings from Joint-Embedding Predictive Architecture (JEPA) encoders better support action generation than compressed VAE latents, with I-JEPA performing best in our encoder comparison. Based on these findings, we propose LeWAM, which conditions action generation on JEPA embeddings and models environment evolution by predicting future embeddings in the same space, without relying on a video diffusion backbone. We further find that imitation learning matches demonstrated actions but does not distinguish better actions from worse ones, even though small action deviations can greatly affect task success. To address this limitation without additional environment interaction or the human oversight required for resets and safety, we introduce Demonstration-Guided DPO (DemoDPO), an offline preference refinement stage that derives preference supervision directly from demonstrations.With only 0.4B trainable parameters, LeWAM achieves an average success rate of 92.28\% on RoboTwin 2.0, comparable to that of state-of-the-art VLAs and WAMs, and maintains practical effectiveness on real-world manipulation tasks.

### 论文解读
#### 摘要翻译
LeWAM 面向机器人世界动作建模，指出视频扩散模型计算昂贵、VAE 潜变量未必保留动作信息。方法改用冻结 I-JEPA 表示，同时预测动作和未来视觉嵌入，并用无需环境交互的 DemoDPO 做离线偏好优化。模型只有约 0.4B 可训练参数，在 RoboTwin 2.0 达到 92.28% 成功率。

#### 方法动机分析
VLA 从观测直接输出动作，缺少世界演化约束；传统 WAM 又依赖重型视频生成。论文认为，VAE 重建偏好高方差像素方向，可能忽略低方差但决定动作的因素。JEPA 的预测式表示因此更适合动作条件与未来状态建模。
核心假设是：只要表示保留与动作相关的因素，未来嵌入预测就能提供有用的物理演化约束，同时避免重建全部像素。本文重点验证视觉表示、效率与离线精炼，尚未充分覆盖自然语言任务和跨平台泛化。

#### 方法设计详解
输入为头部及两只腕部相机图像和任务 ID。冻结 I-JEPA-H/14 将图像变成 256 个 token，AdaFuse 自适应融合编码器多层特征。单个 12 层、隐藏维 1280 的 DiT 风格预测器接收当前嵌入、带噪动作 token 和未来嵌入查询：用 flow matching 生成动作块，同时在 JEPA 空间回归未来嵌入。训练样本每次采样一个未来偏移，推理时去掉未来查询，仅保留动作路径。DemoDPO 从冻结参考策略采样 4 个候选，以与示范的 MSE 排出优劣，差距超过 10^-3 才形成偏好对，再用速度预测误差的相对分数优化策略。

#### 方法对比分析
LeWAM 直接预测 JEPA 嵌入演化，无需生成未来视频；AdaFuse 利用多层视觉信息，世界模型为动作提供未来约束，DemoDPO 则补足行为克隆无法区分候选质量的问题。它适合多视角、离线示范充分的双臂操作；开放语言指令和跨机器人泛化仍需额外验证。
与 VAE 潜变量相比，I-JEPA 更关注预测结构而非像素重建；与视频扩散 WAM 相比，LeWAM 把世界模型目标压缩为嵌入回归；与普通 flow-matching 策略相比，它增加未来状态监督和偏好排序。因此创新点是表示空间、统一预测器和 DemoDPO 的组合，而非重新设计基础流匹配。

#### 实验分析（精简版）
在 50 个 RoboTwin 2.0 双臂任务上，LeWAM 总成功率 92.28%（干净 93.14%、随机化 91.42%），高于 LaWAM 的 91.22% 和 Fast-WAM 的 91.83%。消融中动作模型为 86.17%，加入标准世界模型为 87.02%，再加 AdaFuse 为 90.69%，DemoDPO 后升至 92.28%，说明各模块具有递进收益。H800、batch=1、10 步去噪时，CUDA Graph 延迟 31.90 ms、峰值显存 1.988 GiB。真实双 Piper 实验中，DemoDPO 将堆叠积木、叠毛巾、插花进度分别由 81.1%、58.3%、50.7% 提升到 84.9%、60.7%、53.0%。局限在于模拟任务使用离散任务 ID，实验规模和表示选择仍依赖特定机器人数据。

#### 实用指南
论文给出 PyTorch、CUDA Graph、AdamW 和学习率 5×10^-5 等信息，但未明确确认代码或权重公开。复现需准备三路相机、224×224 I-JEPA 输入、动作分块、未来偏移和 RoboTwin 的干净/随机化示范。迁移时需重做相机与动作接口、任务上下文和示范训练；冻结编码器是否适配新机器人应通过表示探针检查。

#### 总结
JEPA潜变量统一动作世界演化
1. 编码多视角观测并融合多层特征。
2. 联合生成动作、预测未来嵌入。
3. 用示范偏好离线精炼策略。
4. 移除未来查询进行快速控制。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27455v1)
- [arXiv](https://arxiv.org/abs/2609.27455v1)

---

<a id='2609.27656v1'></a>
## [InternW0: A Foundational Physical World Model for Efficient Real-World Interactions](https://arxiv.org/abs/2609.27656v1)

**Authors:** Jisong Cai, Yao Mu, Ganlin Yang, Zhe Cao, Zhangzheng Tu, Xing Gao, Kailin Li, Xinyu Zhan, Lixin Yang, Yangkun Zhu, Haoxiang Ma, Ming Zhou, Qiaojun Yu, Yufei Xue, Liqun He, Yifei Yao, Yifan Zhu, Long Ling, Bingqi Jiang, Haoyu Guo, Xueyue Zhu, Bowen Zhou, Bin Zhao, Tianfan Xue, Chunhua Shen, Weinan Zhang

**Published:** 2026-09-23

**Categories:** cs.RO, cs.AI

**Abstract:**

Physical intelligence requires more than predicting how the world may evolve: predictions must remain actionable as the world continues to change. We introduce InternW0, the first instantiation of the InternW physical world model series from Shanghai AI Laboratory, built around omnimodal interfaces, asynchronous multi-frequency processing, and local physical modeling under partial observations and external influences. InternW0 jointly learns future visual dynamics and continuous robot control through an asymmetric video--action architecture with flow matching. A high-capacity video expert provides longer-horizon predictive context, while a lightweight action expert operates at a faster timescale. Instead of regenerating the future for every action update, InternW0 reuses layerwise K/V and adapts it to newly observed states through observation-conditioned context routing. Domain-specific interfaces and soft prompts support heterogeneous embodiments, while contact-aware post-training incorporates force and tactile signals for contact-rich manipulation. We train InternW0 on approximately 7,200 hours of heterogeneous robot and egocentric data, including EgoLab, a 275-hour real-laboratory egocentric dataset. Evaluation spans simulation benchmarks and real-world scientific tasks, including a 15-stage metal--organic framework synthesis workflow and 5-stage contact- and force-aware dexterous manipulation for general-purpose quantitative pipetting. These results advance scalable, asynchronous, and science-native physical world models for universal and efficient real-world interactions.

### 论文解读
#### 摘要翻译
InternW0 是面向真实世界交互的物理世界模型，结合全模态接口、异步多频率处理和局部物理建模。它以流匹配同时学习未来视觉动力学与连续机器人控制：高容量视频专家负责长程预测，轻量动作专家负责高速控制，并通过观察条件路由复用视觉 K/V 缓存，实现预测与动作执行解耦。

#### 方法动机分析
同步更新视频生成和动作策略会带来高延迟，难以满足机器人闭环控制；不同机器人形态、动作空间以及力觉信号又增加统一建模难度。论文假设慢速视觉预测能提供可复用的局部物理上下文，而动作策略能在新观测到来时独立快速修正。

#### 方法设计详解
输入包括多视图 RGB、语言、本体感受和可选力/触觉。Wan VAE 与 DINOv3 提取视觉特征后，Video DiT 预测未来视频潜变量并缓存逐层 K/V。新观测送入 Chunk K/V Editor，通过注意力编辑缓存，避免每次重跑完整视频模型；Action DiT 再读取编辑后的上下文、最新感官反馈和 soft prompt，输出末端/关节/夹爪等连续动作，并可预测交互力矩。不同机器人映射到 37 维统一接口。训练采用流匹配，在真实目标与高斯噪声之间学习速度场，并联合视频损失和动作关节损失；数据统一为 15 fps，使用 BF16 与 FSDP2，规模约 7,200 小时。推理部署在 RTX 5090D 上，关键路径延迟 60.73 ms，闭环更新频率 16.47 Hz。

#### 方法对比分析
本质创新是“非对称专家+缓存路由”：视频专家慢速提供未来上下文，动作专家在同一上下文上高频更新。相比纯动作策略，它具备视觉动力学预测；相比同步视频—动作模型，它减少实时循环中的昂贵计算；统一动作接口和 soft prompt 便于跨机器人迁移。适用重点是短中程操作，长程逻辑规划和故障鲁棒性仍有限。

#### 实验分析（精简版）
LIBERO 平均成功率达到 98.6%，RoboTwin 2.0-Full 达到 93.12%。在 15 阶段金属有机框架合成流程中，进度率为 68.4%，基准为 10.7%；关键路径延迟 60.73 ms，闭环更新频率 16.47 Hz。增加 50 小时 EgoLab 数据后，RoboTwin Randomized 成功率从 13.73% 提升到 21.97%，说明真实操作数据有助于泛化。论文仍缺少对极端传感器故障和超长规划的充分验证。

#### 实用指南
论文提供项目主页 https://internrobotics.github.io/internw0，但无法确认代码、权重和数据的完整开放范围。复现需对齐多模态数据和 15 fps 时间基准，实现视频/动作双专家、K/V 编辑及 37 维动作映射。部署时让视频专家后台运行、动作专家进入实时控制线程；迁移新机器人时替换动作映射，并针对形态和力觉传感器重新适配 soft prompt 或后训练分支。

#### 总结
核心思想：异步视觉预测加速机器人控制
1. 编码多模态观测。
2. 视频专家预测并缓存上下文。
3. 新观测编辑 K/V 缓存。
4. 动作专家高速输出控制与力矩。
5. 统一接口迁移到不同机器人。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27656v1)
- [arXiv](https://arxiv.org/abs/2609.27656v1)

---

<a id='2609.27467v1'></a>
## [Kairos: Grounded Forecasting of Presence and Directional Flow in 4D Scene Graphs](https://arxiv.org/abs/2609.27467v1)

**Authors:** Iacopo Catalano, Julio A. Placed, Javier Civera, Jorge Peña Queralta

**Published:** 2026-09-23

**Categories:** cs.RO, cs.AI

**Abstract:**

Long-term autonomy in human-populated environments requires anticipating whether and how people will move at times a robot has not yet observed. Existing representations of pedestrian motion face a tradeoff: they either forecast future activity, reducing each location to a scalar rate, or model the full directional distribution, holding it fixed in time. We present Kairos, a predictive directional-flow memory that extends a hierarchical 3D scene graph (3DSG) to a 4D scene graph (4DSG). Every observed voxel of the reconstructed geometry stores a directional mixture and a presence rate, and spectral predictors forecast, for any future query time, both the probability that people are present and the full directional distribution of their motion. Pairwise flow dependence between adjacent voxels supports conditional queries, and per-voxel predictive variances yield calibrated credible intervals that tighten as observations accumulate. We evaluate Kairos on three real pedestrian environments: a robot-collected campus dataset, a shopping mall, and a station concourse recorded continuously for eleven months. Its learned state remains consistent under loop-closure corrections, and its forecasts are competitive with dedicated occupancy and flow models trained on the full detection stream, although Kairos learns from only the small fraction available to a patrolling robot. Finally, we validate the representation on a downstream encounter-probability planning task, where plans computed over the Kairos forecasts encounter more people than plans computed over any time-invariant map at an equal success rate. We provide the code at https://github.com/IacopomC/kairos.

### 论文解读
#### 摘要翻译
长期在人类环境中运行的机器人，需要预测未来哪里有人以及人往哪个方向移动。Kairos把分层3D场景图扩展为4D场景图：每个几何体素保存存在率和方向混合，频谱预测器随未来时间输出存在概率与完整方向分布。相邻体素的流依赖支持条件查询，预测方差还能给出逐渐收紧的可信区间。实验覆盖校园机器人、商场和连续记录11个月的车站大厅；即使只有巡逻机器人获得的少量检测，结果也能与完整检测训练的专用模型竞争，并改善下游相遇概率规划。

#### 方法动机分析
占据图只能回答“是否拥挤”，时间不变的方向图又无法表达早晚高峰反向流动；抽象网格或节点邻近绑定也会在回环校正后错位。论文假设人流包含可由日、周等周期解释的重复部分，随机残差不外推。目标是同时满足方向性、时间条件性和几何接地；因此它不适合完全突发、永久改变的流动，也不建模静止人的航向。

#### 方法设计详解
输入为检测时间、位置和速度。位置挂接机器人TSDF活动体素，速度投影到水平面得到航向与速度。每个体素使用固定8个航向槽的半包装高斯混合，槽间隔π/4，共享航向标准差0.4 rad、速度标准差0.3 m/s；检测按责任软分配，槽权重在线更新，速度均值按体素统计。每个权重和存在率都是标量时间序列，由非均匀频谱预测器从稀疏不规则观测外推，并用预测误差门控选择频谱阶数；推理时按查询时刻直接评估频谱，不需保存完整检测历史。存在通道采用Gamma-Poisson后验，先验参数α0=β0=1，再结合可见帧和检测更新；结合平均速度与体素尺寸估计穿越时间，以队列模型计算未来窗口出现至少一人的概率。邻域证据会收缩稀疏体素的权重，相邻体素还用互信息学习成对流依赖。体素预测按占据强度聚合到导航节点；回环校正时体素重键，平面旋转使方向槽置换或插值，重合体素按样本量池化。

#### 方法对比分析
相较只预测标量的FreMEn、STeF-Map和Aion，Kairos的创新贡献是预测连续的完整方向—速度分布；相较时间不变的CLiFF-map、Rheos，它优于静态平均的关键在于能按查询时刻改变流向；相较坐标场方法，它把状态绑定实际重建体素，能跟随SLAM校正。固定8槽提升跨地点可比性，成对耦合提升邻域相干性，但需要足够观测证据。

#### 实验分析（精简版）
TBD约213分钟机器人数据用于稀疏具身测试；ATC按84/8天、HB按227/28天训练测试，统一采用0.4 m体素。ATC中时间权重相对静态权重使联合MLPD提升0.0076 nats（具身）和0.0115 nats（全可见）；HB对应提升0.0117和0.0139 nats。HB相似邻接体素的成对学习带来+0.63联合MLPD和56.5°条件航向误差改善。权重可信区间在HB的68%/95%覆盖率为0.699/0.964，ECE为0.011，显示不确定性校准较好；低覆盖和突发变化仍是主要边界。

#### 实用指南
论文提供GitHub代码。复现需准备TSDF接地、导航节点支持集、0.4 m体素、8个方向槽、场景候选周期和在线频谱门控；必须把空的可见帧纳入存在率更新。迁移时需替换检测与场景图接口，并在新环境重新估计速度、频谱、邻接耦合和Gamma-Poisson状态。含非平面地图旋转的系统还需扩展当前二维航向表示。

#### 总结
核心思想：时间地图预测接地人流
1. 检测进入TSDF体素，分离存在与方向。
2. 频谱预测8槽权重，邻域证据补足稀疏观测。
3. 成对耦合并聚合到导航图，回答未来流向。
4. 回环时同步重键和旋转，服务规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27467v1)
- [arXiv](https://arxiv.org/abs/2609.27467v1)

---

<a id='2609.27381v1'></a>
## [CoPRE: Improving Sensitivity in Proprioceptive Contact Detection for Low-Cost Robot Arms](https://arxiv.org/abs/2609.27381v1)

**Authors:** Yuxiao Zhu, Jinzhou Li, Yifei Dong, Muhammad Suhail, Chunyuan Yang, Xinyuan Luo, Haoyu Li, Boyuan Chen, Xianyi Cheng

**Published:** 2026-09-23

**Categories:** cs.RO

**Abstract:**

Contact detection during robotic manipulation allows robots to recognize unexpected contact and adapt their motion accordingly. However, in low-cost robot arms without dedicated force or tactile sensors, detecting weak contacts from proprioception is challenging because the resulting changes in joint-level proprioceptive signals can be small compared to normal variation and noise caused by robot motion itself. We introduce Contact-free Proprioceptive Response Estimation (CoPRE), improving proprioceptive contact detection sensitivity using only contact-free motion, without additional force sensors, contact labels, or analytical dynamics models. CoPRE estimate the expected joint torques under contact-free motion from proprioceptive state history and commanded motion, while removing recent observations that may already reflect contact. It then computes the residual between the expected and observed joint torque estimates, and maps this residual to a contact score using a noise-weighted Jacobian. Real-robot experiments on ARX Arm and Unitree G1 show that CoPRE achieves 74.1% and 82.2% recall on the tested contact trials, compared with 0%/0% on ARX and 16.3%/42.2% on G1 for the learned torque-prediction and inverse-dynamics baselines. CoPRE also reaches 90% detection rate for pushing force at 3.5 N on ARX and 5.5 N on G1. To demonstrate the downstream utility of our method, we implement belief-space manipulation planning for obstacle-aware object placement and book insertion where detected contacts update the spatial belief and enable the robot to retreat from blocked motions, adjust its pose, and retry. Project website at https://copre-arm.github.io

### 论文解读

#### 摘要翻译
CoPRE 面向没有力/触觉传感器的低成本机械臂，只利用关节位置、速度和电机力矩等本体感知信号检测弱接触。它先预测无接触时的正常响应，再从观测与预测的差异中估计接触力，并用时间一致性减少误报。在 ARX L5 六自由度机械臂和 Unitree G1 七自由度手臂上的实验表明，该方法能感知较弱的外力，并用于接触感知操作。

#### 方法动机分析
运动噪声、摩擦和控制波动常常淹没弱接触信号，直接阈值检测会在灵敏度与误报之间折中。普通预测器若看到接触后的最新状态，可能把扰动当成正常运动并追随它，使残差变小。CoPRE 的关键假设是：动作命令可以看到当前时刻，但状态历史应在接触前截断；这样模型保持无接触基准，接触变化才会显现。

#### 方法设计详解
状态输入包含关节位置、速度和观测力矩，命令包含期望位置与速度。Transformer 编码器—解码器使用长度 L=20 的历史窗口，并排除最近 H=3 个状态步；它据此预测最近时段的名义状态。训练损失以力矩误差为主，位置和速度误差各占 0.1，网络为 96 维、4 个注意力头、2 层编码器和解码器，采用 AdamW，学习率 5×10^-4。推理时计算观测力矩与预测力矩的残差，减去偏置后，用由无接触数据 MAD 得到的噪声尺度加权，并通过当前雅可比的阻尼最小二乘反演等效末端力。力的范数作为接触分数，连续 K=3 个采样超过阈值才报警。

#### 方法对比分析
相比 LSTM 预测基线 NEXT，CoPRE 的本质创新是状态排除，避免预测器吸收接触扰动；相比标称逆动力学 Dynamics，它学习更贴近实际控制和摩擦的无接触响应。噪声加权让波动大的关节少影响判断，连续确认则抑制瞬时尖峰。方法适合能获得关节状态、电机信号并计算雅可比的机械臂，但更换机器人、负载或控制器时需要重新校准。

#### 实验分析（精简版）
实验推动不同重量的书堆，产生约 1.5–5.5 N 阻力。在 ARX 上，CoPRE Recall 为 74.1%、F90 为 3.5 N，而 NEXT 与 Dynamics 在测试范围内 Recall 均为 0%；在 G1 上，CoPRE Recall 为 82.2%，高于 Dynamics 的 42.2% 和 NEXT 的 16.3%。消融表明取消状态排除后，ARX 召回率降至 40% 以下。代价是需要机器人专属无接触数据，且 K=3 带来约 60–100 ms 的确认延迟。

#### 实用指南
论文给出项目网页 copre-arm.github.io，包含演示和资料；独立代码仓库或预训练模型的发布状态未明确说明。复现需采集无接触运动数据，计算一致的力矩、雅可比和偏置，使用 MAD 标定关节噪声，再调节阈值与连续步数。迁移到新机械臂时要更换接口和雅可比，重新训练名义预测器；负载变化还应单独测试弱力 Recall、误报率和检测延迟。

#### 总结
核心思想：排除接触状态再加权残差

1. 用延迟状态预测正常响应。
2. 按噪声尺度加权力矩残差。
3. 用雅可比反演等效接触力。
4. 连续超阈值后确认接触。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27381v1)
- [arXiv](https://arxiv.org/abs/2609.27381v1)

---

<a id='2609.27450v1'></a>
## [BEE: Intervention-Adaptive Real-World Reinforcement Learning with Vision-Language-Action Models](https://arxiv.org/abs/2609.27450v1)

**Authors:** Weihui Zhao, Xiaohan Yan, Zunian Wan, Xuan Du, Zhaozhan Chi, Jianbo Mao, Ruipu Wu, Rushuai Yang, Houlin Li, Shukai Yang, Jing Wu, Yuxiang Yan, Yongcheng Liu, Chuankang Li, Guanghui Ren, Wei Shan, Maoqing Yao

**Published:** 2026-09-23

**Categories:** cs.RO, cs.AI

**Abstract:**

Vision-language-action (VLA) models handle long-horizon manipulation, yet success hinges on a few precision-critical phases where millimeter-scale errors undo all prior progress. Online reinforcement learning (RL) can optimize exactly these actions, but free exploration is far too costly on real robots, which makes human corrections indispensable. However, existing online RL methods for VLAs either cannot incorporate such corrections or fold them into undifferentiated supervision. Yet human corrections are not uniformly noisy but reliable along some action dimensions and variable along others. Building on this, we introduce BEE, an intervention-adaptive framework for real-world RL on a frozen VLA that lets the policy go BEyond Expert imitation. We formulate human corrections not as actions to reproduce but as evidence about a constraint: a Correction Model predicts how a human would correct a given VLA proposal and how consistent the correction is along each action dimension. This predicted consistency sets the per-dimension tightness of a constraint on policy optimization. Where corrections are consistent the policy stays close to the human, and where they vary, the constraint relaxes. We evaluate BEE on three real-world manipulation tasks and one LIBERO-Pro simulation task at a matched online-data budget. BEE attains the highest success rate on every task, 91.2% on average against 57.5% for RLT and 42.1% for DSRL, and the lowest human intervention rate on all real-world tasks.

### 论文解读
#### 摘要翻译
视觉-语言-动作（VLA）模型擅长通用操作，却常在插入、对位等精密阶段因毫米级误差失败。真实机器人在线强化学习能修正这些问题，但自由探索昂贵且危险。BEE（BEyond Expert imitation）学习人类相对于 VLA 动作提议的修正分布，把干预转化为不确定性感知的策略约束，在探索与安全之间取得平衡。

#### 方法动机分析
传统人类在环方法多把纠正当作单一动作标签或标量置信度，无法区分不同动作维度的可靠性。例如人类对平移修正很一致，对旋转修正却可能有分歧。BEE 假设修正分布可以从少量干预中学习，并用逐维方差决定约束强弱：一致维度严格遵循人类，不确定维度允许策略探索。

#### 方法设计详解
冻结 VLA 从图像、语言和机器人状态产生潜在 token 及动作提议；轻量残差策略输出 Δ，并将最终动作写成 a=a~+Δ。Correction Model 用人类纠正数据学习高斯分布，输出修正均值和对角协方差。策略同时追求评论家 Q 值、靠近 VLA 提议，并限制与预测纠正动作的归一化 Mahalanobis 距离。低方差维度的偏离会受到更大惩罚，高方差维度则保留探索空间；状态相关拉格朗日乘子进一步调整不同状态的约束强度。推理时冻结 VLA，逐状态提取 token 和本体感知，计算残差后合成最终动作。实验使用 20 个预收集纠正回合和 70 个在线真实机器人回合。

#### 方法对比分析
BEE 不把人类动作硬编码成唯一目标，而是建模其均值与可靠性；相比只使用状态级不确定性的方案，它保留了动作空间的逐维几何信息；相比直接在线 RL，它以 VLA 和人类分布限制危险探索。因此它尤其适合 VLA 能完成粗动作、却在接触和插入阶段失败的任务。

#### 实验分析（精简版）
真实任务包括手机充电、零食挂架和布料对齐，并在 LIBERO-PRO 碗放置上测试。BEE 平均成功率为 91.2%，高于 RLT 的 57.5% 和 DSRL 的 42.1%；手机充电达到 100%，零食挂架达到 85%，而基础策略在零食挂架上为 0%。手机充电干预率为 12.2%，RLT 为 22.5%。去掉 Mahalanobis 逐维约束后，零食挂架成功率约降至 30%，说明可靠性建模确实重要。任务规模和 VLA 初始能力仍限制结论外推。

#### 实用指南
复现需保留 VLA 的动作提议，收集时间对齐的人类纠正，训练高斯 Correction Model，再优化残差策略。应统一平移、旋转等动作维度的尺度，并同时报告成功率与干预率。迁移到新机器人时要替换本体感知、动作范围和控制频率，重新收集纠正数据并调节约束阈值；论文未明确确认代码、模型或数据已公开。

#### 总结
核心思想：按维度约束人类修正
1. VLA 提议动作。
2. 学习人类纠正的均值和方差。
3. 用 Mahalanobis 距离约束残差探索。
4. 通过状态相关乘子在线改进。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27450v1)
- [arXiv](https://arxiv.org/abs/2609.27450v1)

---

<a id='2609.27363v1'></a>
## [From LiDAR Maps to Visual Localization: Unified Visual Association for Robust Point-Line-Plane Pose Estimation](https://arxiv.org/abs/2609.27363v1)

**Authors:** Wentao Zhao, Zikun Chen, Yihe Niu, Haoyu Chen, Jingchuan Wang

**Published:** 2026-09-23

**Categories:** cs.RO

**Abstract:**

Camera localization in a prior LiDAR map provides a persistent geometric reference for long-term robotic navigation, yet remains challenging because of the substantial modality gap between camera images and point-cloud maps. We present a unified localization framework that makes the LiDAR map visually addressable rather than relying on a dedicated image-LiDAR correspondence model. Map geometry and reflectivity are rendered into LiDAR-derived quasi-images with explicit 2D-3D provenance, enabling camera observations and rendered map views to share mature visual features and matchers for both global localization and continuous pose tracking. Point and line correspondences are established through this common visual interface, while the retained provenance recovers metric LiDAR geometry and line-supported planar constraints for pose estimation. To improve robustness under ambiguous associations and weak geometry, we further introduce a distribution-aware, observability-complementary optimization strategy. Instead of reducing matching ambiguity to a scalar confidence, candidate association distributions are propagated into directional pose-information uncertainty, and reliable structural factors are selectively reinforced according to their ability to complement the currently weak pose directions. Experiments on the EuRoC MAV benchmark and self-collected real-world sequences demonstrate accurate global localization and robust continuous 6-DoF tracking using only a pre-built LiDAR map as the persistent prior, including under severe illumination variations and dynamic occlusions.

### 论文解读
#### 摘要翻译
本文提出统一定位框架，让 LiDAR 地图具备“视觉可寻址性”。地图几何与反射率被渲染为准图像，因此相机图像和地图可以共享成熟视觉特征与匹配器。框架支持全局定位和连续六自由度跟踪，并以点、线、平面约束增强稳定性；优化器会根据当前位姿信息中的薄弱方向，优先增强能补足这些方向的结构因子。

#### 方法动机分析
相机外观会随光照、视角和遮挡变化，难以作为长期地图锚点；像素与三维点的直接跨模态匹配又缺少统一表征。即使匹配很可靠，也可能集中在同一几何方向，例如走廊中难以约束纵向运动。论文因此假设已有度量 LiDAR 地图，并希望同时解决关联能力和位姿可观测性问题。

#### 方法设计详解
系统输入 LiDAR 地图、双目图像和 IMU。它从预测视点渲染局部地图，把反射率编码到色相，把深度及其边界编码到明度，得到带二维位置、深度和三维来源的准图像。真实图像与准图像都送入 PL-Net 提取点、线特征，再建立跨域匹配。全局阶段从离线准图像库检索初始位姿；跟踪阶段利用 IMU 和上一帧预测位姿，生成局部准图像并构造点、线、平面（PLP）几何残差。系统保留候选匹配分布及测量权重，不只使用单个最佳匹配。随后分析信息矩阵特征值，用弱性系数识别欠约束的六维位姿方向，并按照每个因子对这些方向的补足程度重加权。直观上，优化器会选择“当前最需要的证据”。论文报告在 Ryzen 9 7945HX、RTX 4060 笔记本上跟踪约 13.3 Hz。

#### 方法对比分析
相较 RGB 外观地图，准图像依赖持久的 LiDAR 几何和反射率，降低光照变化影响；相较仅使用几何原语的方法，统一视觉前端提供更强的点线关联，同时保留精确的二维—三维来源。其关键差异是观测性互补加权：高置信匹配若无法约束当前薄弱方向，也不会被盲目放大。方法更适合具有墙面、边缘和稳定反射结构的室内或结构化场景。

#### 实验分析（精简版）
在 EuRoC MAV 及自采集的光照变化、动态遮挡序列上，完整方法的全局定位 recall 为 88.2%，平均 ATE 为 8.09 cm。连续跟踪在 V201 和 V202 上的 ATE RMSE 分别为 1.4 cm 和 1.5 cm。点线面联合的 ATE 为 2.35 cm，点-only 为 3.74 cm；观测性与分布建模使 EuRoC 结果由均匀权重的 5.08 cm 降至 2.35 cm，真实数据由 55.21 cm 降至 24.27 cm。自采集困难光照下 ATE 为 16.0 cm，TC-VIML 为 275.8 cm。局限是地图退化或结构稀疏时线面约束减少。

#### 实用指南
论文使用 EuRoC 和自采集数据，正文未给出代码公开链接。复现需准备度量 LiDAR 地图、双目与 IMU 标定、准图像数据库，以及每个视觉元素的三维来源；重点实现反射率/深度编码、PL-Net 匹配、PLP 残差和信息矩阵重加权。迁移到其他机器人时需要重建地图视点数据库，替换标定与投影模型，并依据新环境的结构分布重新调节线面提取和权重参数。

#### 总结
核心思想：按观测缺口融合视觉与 LiDAR
1. 把 LiDAR 地图渲染成可匹配的准图像。
2. 用统一点线特征建立二维—三维关联。
3. 形成点线面几何约束并保留匹配分布。
4. 按位姿薄弱方向重加权，输出稳健定位与跟踪。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27363v1)
- [arXiv](https://arxiv.org/abs/2609.27363v1)

---

<a id='2609.27513v1'></a>
## [Behavior-Aligned Action Tokenization for Robot Policy Learning](https://arxiv.org/abs/2609.27513v1)

**Authors:** Junbo Dong, Ze Chen, Zhendong Xie, Junjie Li, Lixin Xu, Xuemin Chi, Yiming Song, Zhaoyuan Ma

**Published:** 2026-09-23

**Categories:** cs.RO

**Abstract:**

Autoregressive robot policies learn continuous control by predicting discrete action tokens from observations. Different tasks often share local motions, yet behavioral correspondence across demonstrations receives limited explicit supervision in existing tokenizers. Motions with different timing can therefore lack a shared representation despite following similar patterns. We propose Behavior-Aligned Action Tokenization (BAAT), which uses soft dynamic time warping (Soft-DTW) to select corresponding action chunks and aligns their quantized coordinates jointly with reconstruction. This objective encourages similar motions across tasks to occupy nearby quantized representations while retaining executable action detail. A history-conditioned diffusion decoder reconstructs continuous action chunks from these tokens, and a downstream autoregressive policy learns to predict them. We evaluate BAAT on selected tasks from three simulation benchmarks and two real robot tasks. BAAT achieves a mean simulation success rate of approximately 45.2%, exceeding OAT by approximately 7.2 percentage points. In the controlled LIBERO-All alignment ablation, policy success rises from 70.2% to 79.0% while trajectory replay success decreases. These results support behavioral correspondence as supervision for organizing shared motion structure in action tokenizers and improving downstream robot policy learning.

### 论文解读
#### 摘要翻译
BAAT（Behavior-Aligned Action Tokenization）为机器人策略学习提出行为对齐的动作标记化方法。它针对现有离散动作 tokenizer 只优化重建、难以让跨任务的相似行为共享表示的问题，引入 Soft-DTW 对齐时间错位的动作片段，并在量化空间中约束它们接近；随后用自回归策略预测离散动作 token。

#### 方法动机分析
抓取、抬升等局部技能可能以不同速度执行。逐点距离会把它们误判为不同动作，单独的重建损失也不会自动形成可共享的动作邻域。BAAT 的假设是：Soft-DTW 找到的低距离片段在行为上相似，将其 token 对齐后，多任务策略更容易学习共享结构；但对齐过强会损失动作细节，也需要足够多样的演示数据支撑。

#### 方法设计详解
连续动作块先经过编码器，再由 FSQ 量化为 16 个 latent slots，每个 slot 含 4 个标量因子，量化级别为（8,5,5,5）。训练时在 mini-batch 内用 Soft-DTW 选出相似动作对，对其量化坐标施加平方距离损失，并与带掩码的 Smooth-L1 重建损失相加，权重 λ=0.1。条件扩散解码器接收量化坐标和历史执行动作，迭代去噪生成长度 H=20 的动作块，实际执行 16 步；历史输入还帮助模型处理相邻动作块的边界连续性。冻结 tokenizer 后，把专家动作转成离散标签，自回归策略根据观察和任务目标以因子化 NLL 预测 token。

#### 方法对比分析
FAST 侧重字节压缩，Bin 直接分箱，OAT 使用排序式标记化，Diffusion Policy 则直接预测连续动作。BAAT 的关键区别是把时间对齐得到的行为关系直接写入量化空间，适合多任务、动作相位差异明显的机器人数据。条件扩散和历史输入主要负责重建质量与边界平滑，行为对齐是方法的核心机制。

#### 实验分析（精简版）
在 LIBERO、RoboCasa、RoboTwin 2.0 及真机任务上，联合训练平均成功率 BAAT 为 45.2%，高于 OAT 的 38.0%、Diffusion Policy 的 34.1% 和 FAST 的 24.4%。LIBERO-All 达到 79.0%，比 OAT 高 2.5 个百分点；从单套件到联合训练提升 8.1 个百分点（70.9%→79.0%）。消融显示无对齐为 70.2%，λ=0.1 为 79.0%，λ=0.5 降至 0%；逐点 L2 匹配为 73.0%，Soft-DTW 为 79.0%。这说明时间对齐和适度正则同时影响泛化收益，调大权重并不会持续改善结果。牙刷整理真机成功率为 70%（14/20）。

#### 实用指南
复现需实现 FSQ、Soft-DTW 配对、条件扩散和历史动作条件，按“先训练 tokenizer、再冻结并训练自回归策略”的顺序执行。论文使用公开基准，但未明确给出官方代码开源状态；除已报告的 H=20、执行 16 步、λ=0.1 外，完整学习率、batch size 和训练轮数未说明，复现时应记录这些缺失配置。迁移到其他机器人需重设动作维度、归一化和历史接口，并重新训练两阶段模型。

#### 总结
核心思想：用时间对齐塑造动作词汇
1. 编码并量化连续动作块。
2. 用 Soft-DTW 对齐相似行为。
3. 让对应 token 接近并由扩散器重建。
4. 训练策略预测对齐后的动作 token。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27513v1)
- [arXiv](https://arxiv.org/abs/2609.27513v1)

---

<a id='2609.27227v1'></a>
## [Surgical Kinematics from Monocular Video with Learned Articulated Motion Constraints](https://arxiv.org/abs/2609.27227v1)

**Authors:** Mehmet Kerem Turkcan, Soham Samal, Zoran Kostic

**Published:** 2026-09-23

**Categories:** cs.CV, cs.RO

**Abstract:**

Objective assessment of robotic surgery uses instrument kinematics, which must be reconstructed when only video is available. We introduce a kinematic reconstruction network for estimating instrument position, orientation and jaw angle from monocular video. Our visual representation combines global attention pooling of frozen DINOv3 features with local pooling at instrument landmarks from fine-tuned SAM 3.1 masks. Our shared Transformer encoder and temporal convolutional heads integrate this representation with mask geometry, monocular depth and visual state estimates from arm-specific multilayer regression networks. Our position branch predicts displacement magnitude and direction separately to preserve traveled distance. We fit trajectories to predicted state observations and motion increments by differentiable weighted least squares, expressing quaternion observations relative to cumulative predicted rotations to obtain a quadratic orientation objective. We evaluate reconstruction across 2,802 Open-H episodes. Compared with LiveMAE on the main Open-H benchmark, our method reduces path-length mean absolute error from 0.45 to 0.34\,cm and increases temporal mean average precision for motion segmentation from 44.54\% to 54.44\%.

### 论文解读
#### 摘要翻译
本文从单目手术视频重建手术机器人运动学，包括器械位置、四元数姿态和夹具角度。方法融合视觉表征、状态条件 Transformer、运动预测与可微轨迹重建，并学习关节运动约束，在 Open-H 数据集和运动分割任务上验证。

#### 方法动机分析
商业手术机器人通常不给出同步运动学接口，而单目视频又存在遮挡、外观变化和深度不可观测。逐帧回归的小位置误差会累积成明显路径长度误差，时间平均还会模糊真实运动。作者据此假设，单帧视觉证据必须与连续运动规律联合使用，才能得到稳定、可解释的轨迹。

#### 方法设计详解
每帧输入为 215 维特征：DINOv3 ViT-L/16 提供全局外观，微调 SAM 3.1 提供器械质心、轴端点、腕部和夹具关键点，并在其周围做局部池化；掩码几何和 EndoSynth Depth Anything V1 ViT-B/14 的关键点相对深度补充结构与深度信息。初始状态回归器先估计姿态和夹具角度，随后由 4 层、4 头的状态条件 Transformer 融合视觉、状态和相对时间。TCN 预测观测、运动增量及可信度权重，幅度—方向分支单独预测位移大小和方向。位置与夹具通过可微加权最小二乘重建；四元数变量替换则把姿态约束转化为可求解的二次目标。训练分视觉、状态细化、位移三阶段进行，并使用三折交叉验证。

#### 方法对比分析
相较 MS-TCN、PatchTST、Transformer Encoder 和 LiveMAE 主要进行时序预测，本文的本质区别是把预测器与可微运动学重建结合，显式建模观测与增量的可信度、位移幅度/方向及四元数连续性。创新包括学习非均匀权重以降低遮挡帧影响、用幅度—方向分支针对路径长度漂移，以及用四元数变量替换保持姿态连续。它适合具有明确器械结构和连续运动约束的手术视频。

#### 实验分析（精简版）
Open-H 包含 2,802 个片段、746,001 帧，并采用三折交叉验证。相较 LiveMAE，路径长度 MAE 从 0.45 cm 降至 0.34 cm，姿态误差从 12.59° 降至 8.78°；运动分割 mAP 从 44.54% 提升至 54.44%，说明重建质量改善能传递到下游技能分析。加入位置细化后 Path MAE 从 0.3854 cm 降至 0.3392 cm。方法仍依赖可靠分割和手术数据微调，跨机器人泛化证据有限。

#### 实用指南
论文使用公开 Open-H 及其子集，并给出三阶段训练、4 个 Transformer block 和 4 个注意力头等设定；完整代码、依赖和硬件信息未在现有内容中明确。复现需保持关键点、掩码几何、相对深度和运动增量定义一致，实现可微 WLS 与四元数变量替换。迁移到其他机器人时应重新训练视觉、关节表示和运动约束，并同时评估位姿与路径长度。

#### 总结
核心思想：学习约束重建手术轨迹

速记：
1. 单目帧提取全局、关键点、几何和深度特征。
2. Transformer 估计状态，TCN 预测观测、增量和权重。
3. 幅度—方向分支保护路径长度。
4. 可微 WLS 与四元数变换输出一致轨迹。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27227v1)
- [arXiv](https://arxiv.org/abs/2609.27227v1)

---

<a id='2609.27346v1'></a>
## [Reflection-Aware Reasoning for Non-Line-of-Sight Pedestrian Localization](https://arxiv.org/abs/2609.27346v1)

**Authors:** Byeonggyu Park, Mingu Jeon, Seong-Woo Kim

**Published:** 2026-09-23

**Categories:** cs.RO

**Abstract:**

Reliable localization of non-line-of-sight (NLOS) pedestrians is critical for safe urban autonomous driving, yet it remains highly challenging in ego-dynamic outdoor environments, where ego-vehicle motion makes radar multipath propagation complex and noisy. In this paper, we present a reflection-aware framework for NLOS pedestrian localization with a moving ego-vehicle in outdoor testbed scenarios. Our framework fuses front-view camera images and 2D radar point clouds to infer reflection orders and reflective surface distributions in bird's-eye-view space. It then uses physics-guided ray tracing to reconstruct distorted reflection paths and localize the hidden pedestrian. We validate the framework in outdoor testbed scenarios under ego-dynamic conditions. The results demonstrate the effectiveness of the proposed framework for NLOS pedestrian localization with a moving ego-vehicle.

### 论文解读
#### 摘要翻译
论文面向自车运动户外环境中的非视距（NLOS）行人定位。方法融合前视相机与二维雷达，在鸟瞰图中推断反射类型和反射面，再以物理引导射线追踪恢复被遮挡行人位置。

#### 方法动机分析
建筑和墙体会遮挡行人；既有雷达方法常依赖固定几何或规则，难适应运动车辆带来的噪声和多径变化。作者假设视觉可提供结构线索，雷达可提供距离与反射证据，二者结合后能学习出可解释的反射中间表示。

#### 方法设计详解
输入为同步前视RGB图像和经自车运动补偿的多帧雷达点云。LSS图像编码器把图像提升到BEV，雷达编码器把累积点云编码到同一空间；交叉注意力令雷达作Query、图像作Key/Value。融合特征一方面逐点分类反射类型，另一方面预测BEV反射面概率图。训练时还加入图像语义分割辅助损失。推理时用DBSCAN聚类一阶、三阶回波；沿雷达原点到三阶聚类中心的射线搜索最可能的反射面，再按平面镜像公式把虚像还原为行人位置，最后合并候选。数据包含120个场景、12539帧，动态车辆最高22 km/h；系统在RTX 3090上约72 ms完成一帧、13.85 FPS。作者将二阶反射排除在几何推理之外，因为这类回波通常对应LOS条件并伴随更可靠的一阶观测；最终定位因此保留了明确的物理解释，而非直接回归坐标。

#### 方法对比分析
本文的区别在于同时学习“哪些雷达点属于何种反射”和“反射面在哪里”，再由物理几何完成定位。相比预先给定墙体或纯规则推理，它减少了场景先验依赖，并面向户外自车运动和多目标场景；但仍依赖可观测的较规则反射面与良好运动补偿。

#### 实验分析（精简版）
在动态条件下，点分类准确率为89.9%、Macro-F1为0.814，NLOS平均定位误差为1.23 m；最近基线为3.23 m，误差降低超过60%。去掉相机后动态反射面IoU从0.738降至0.672，体现视觉结构信息的贡献。局限是测试主要在T形路口线性墙体中进行，复杂反射器可能造成几何歧义。

#### 实用指南
论文声明代码、数据和模型将在项目主页提供，但正文未列具体链接。复现需相机、77 GHz雷达、LiDAR和轮速计，并保持雷达点云的FMCW/CFAR处理、坐标变换和运动补偿一致。迁移到新平台时需重新标定BEV坐标、构造反射面监督并重训融合模型；论文未报告跨数据集泛化结果。

#### 总结
核心思想：学习识别反射，再用物理镜像定位。
1. 多帧雷达与相机BEV交叉融合。
2. 预测反射类型和反射面热力图。
3. 沿射线寻找反射面并镜像三阶回波。
4. 合并候选得到隐藏行人位置。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.27346v1)
- [arXiv](https://arxiv.org/abs/2609.27346v1)

---

