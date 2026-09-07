time: 20260907

# Arxiv Computer Vision Papers - 2026-09-07

## Table of Contents

1. [MINT: A Unified Model for World-Space Camera and Hand Motion Estimation from Scalable Egocentric Pipeline Supervision](#2609.04958v1)
2. [NavArena: Automated Construction of Goal-Oriented Navigation Benchmarks from 3D Gaussian Splatting Reconstructions](#2609.04602v1)
3. [Dressing in Motion: A Human Motion-Aware Diffusion Policy for Robot-Assisted Dressing](#2609.04759v1)
4. [Linguistic Trajectory Encoding for Efficient Long-Horizon Spatial Memory in Embodied Agents](#2609.04802v1)
5. [FIRE-LIVWO: Robust LiDAR-Inertial-Visual-Wheel Odometry via Failure-Immune mmWave Radar Enhancement](#2609.05325v1)
6. [Human-Human & Human-Robot Interaction Transformer (H2INT) for Robot Navigation in Dense and Uncertain Crowds](#2609.05300v1)
7. [One Word, Different Action: A Real-Robot Benchmark for Language-Conditioned Embodied Reasoning](#2609.05260v1)
8. [HiSfM: Disambiguating Structure-from-Motion via Scaffold-Anchored Hierarchical Reconstruction](#2609.04718v1)
9. [LIBERO-RECOVER: Beyond Task Success Towards Failure Recovery in Robotic Manipulation Models](#2609.05178v1)
10. [RoboSPA: Can VLA Models Go Beyond Simple Scenes and Short-Horizon Tasks?](#2609.05324v1)

---

## Papers

<a id='2609.04958v1'></a>
## [MINT: A Unified Model for World-Space Camera and Hand Motion Estimation from Scalable Egocentric Pipeline Supervision](https://arxiv.org/abs/2609.04958v1)

**Authors:** Zijie Zhu, Weiren Cai, Yizhou Wang, Zhenjie Yang, Yide Liu, Jiahao Chen, Guanqi He

**Published:** 2026-09-04

**Categories:** cs.CV, cs.RO

**Abstract:**

Recovering camera and hand motion in world coordinates from egocentric video is a key capability for activity understanding, robot learning, and augmented reality. Existing systems typically decompose this problem into separate stages for camera motion, depth, hand reconstruction, and trajectory refinement, resulting in substantial computational overhead and preventing the joint modeling of camera and hand motion. We introduce MINT (Minting IN-the-Wild Trajectories), the first foundation model that directly produces complete world-space two-hand trajectories from ego-centric RGB video. From a single shared spatiotemporal video representation, MINT jointly predicts the camera trajectory, camera-frame hand states, and per-frame hand presence, and then produces world-space hand motion via explicit coordinate transformations. Training such a model at scale is challenging, since paired world-space camera and hand annotations are scarce. We therefore develop an open-source labeling EGOPIPELINE that converts large collections of public egocentric videos into structured camera-and-hand trajectory supervision. MINT is first pretrained on these large-scale pseudo-labels and then fine-tuned on a small set of high-quality joint annotations. Across public benchmarks, MINT achieves [xxx] improvement in world-space hand trajectory accuracy, [xxx] improvement in camera trajectory estimation, and [xxx] faster end-to-end trajectory generation than the labeling pipeline, while generalizing zero-shot to unseen egocentric datasets. We release the model, training and inference code, labeling pipeline, and a curated 1,021-hour egocentric trajectory dataset.

### 论文解读

#### 摘要翻译

MINT 从自我中心 RGB 视频直接生成世界坐标中的相机与双手轨迹。它用共享的时空表示联合预测相机轨迹、相机坐标手状态和逐帧手可观测性，再通过坐标变换得到世界坐标手运动。为解决联合标注稀缺，作者开发开源 EgoPipeline，把公共视频转成相机—手轨迹监督；模型先用大规模伪标签预训练，再用少量高质量标注微调，并在未见数据集上零样本泛化。

#### 方法动机分析

传统系统把相机、深度、手重建和轨迹修正拆成多个阶段，计算昂贵且误差会级联；相机与手其实共享场景几何和运动线索。专用动捕标注又难覆盖野外视频。论文假设成熟几何模块可以先产生规模化伪标签，而一个联合模型能把相机—手耦合关系摊销为单次前向推理。

#### 方法设计详解

EgoPipeline 先用手检测筛选视频，再由 GeoCalib、MoGe-2、MegaSaM/DROID-SLAM 和 HaWoR 分别估计内参、深度、相机姿态和相机坐标手状态，经过离群剔除、SLERP 插值、UKF 平滑后组合到世界坐标，形成 1,021 小时监督。MINT 输入 32 帧、378×518 RGB，采用 ViT-L/14 DINOv2 的几何编码器与交替 Frame/Global Attention 的聚合器。Camera Head 迭代 4 次回归 SE(3) 与 FoV；MANO Head 解码双手 109 维状态；Observability Head 预测手是否可见。通过 p^w=R^T(p^c−t)、Q^w=R^TQ^c 显式变换坐标，并联合优化相机、MANO、可观测性和世界一致性损失。训练分伪标签预训练与高质量标注微调两阶段，使用 AdamW、BF16；推理以滑窗处理，并在重叠处做 SE(3) 对齐拼接。

#### 方法对比分析

MINT 的本质区别是把相机和双手作为一个几何耦合任务直接预测，而不是串联多个专用模型。EgoPipeline 负责扩大监督规模，MINT 负责把昂贵的重建过程摊销为高速前向；UKF 和窗口对齐仍是工程配套。它适合活动理解、机器人学习和增强现实，但单目尺度与长程漂移使成熟全局 SLAM 仍有优势。

#### 实验分析（精简版）

在 HOT3D（27 个序列、94,978 帧）上，MINT+UKF 的手检测 FAcc 为 0.940，超过 WiLoR 的 0.827；MPJPE-p 为 23.62 mm，优于 30.97 mm。ARCTIC 上相机 RPE-T 为 3.39 mm，优于 MegaSaM 的 8.73 mm，但 ATE 为 181.7 mm，高于 DROID-SLAM 的 49.1 mm。混合数据将 MPJPE-p 降至 23.61 mm，UKF 将抖动从 11.52 降至 2.39 mm/f²（约 79%）。这些结果说明统一建模对局部手部和相机相对运动有效，但不能据此宣称全局定位全面超越传统系统。主要局限是伪标签误差、单目尺度漂移和长序列累积漂移。

#### 实用指南

论文提供模型、训练/推理代码、EgoPipeline 和 1,021 小时数据集。复现时需严格统一内参、相机—世界坐标变换、时间同步和 MANO 参数定义，保持 32 帧窗口与 378×518 输入，并实现两阶段损失和重叠窗口 SE(3) 对齐。迁移到新相机或机器人时，应重做标定，并用少量高质量联合标注微调；评估不能只看局部相对误差，还要检查全局 ATE。

#### 总结

核心思想：统一预测相机与双手世界轨迹

1. 几何流水线把野外视频转成相机—手伪标签。
2. 时空聚合器联合回归相机、MANO 与可观测性。
3. 显式坐标变换施加世界轨迹一致性。
4. 滑窗对齐拼接长轨迹并用 UKF 稳定手运动。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.04958v1)
- [arXiv](https://arxiv.org/abs/2609.04958v1)

---

<a id='2609.04602v1'></a>
## [NavArena: Automated Construction of Goal-Oriented Navigation Benchmarks from 3D Gaussian Splatting Reconstructions](https://arxiv.org/abs/2609.04602v1)

**Authors:** Junhui Wang, Wei Yang, Xinyao Li, Ningjing Fan, Yuehao Yin, Xuecheng Chen, Chao Gao

**Published:** 2026-09-04

**Categories:** cs.RO

**Abstract:**

Fixed 3D Gaussian Splatting (3DGS) reconstructions provide realistic novel views but lack the traversability constraints, valid goals, and closed-loop protocols required for navigation evaluation. We introduce NavArena, an automated framework that transforms fixed 3DGS reconstructions into benchmarks for goal-oriented visual navigation. NavArena integrates a frozen 3DGS model for egocentric RGB-D rendering, an occupancy costmap derived from Gaussian density and height statistics for reachability and collision queries, and semantic goal candidates lifted from multi-view open-vocabulary masks. These components support the automatic generation and unified closed-loop evaluation of goal-oriented navigation episodes. Across more than 2{,}000 scenes, NavArena generates 22.2 million expert trajectories. Spatial and semantic evaluations assess the derived navigation representations, while policy rollouts demonstrate the diagnostic value of the unified evaluation protocol. NavArena enables scalable and reproducible navigation evaluation on large-scale 3DGS reconstructions, and all benchmark-generation tools, evaluation protocols, and derived assets will be released publicly.

### 论文解读

#### 摘要翻译

固定3D Gaussian Splatting（3DGS）重建能生成逼真新视角，却没有导航评估所需的可通行性、有效目标和闭环协议。NavArena把冻结3DGS、占用代价图和多视角开放词汇语义结合，自动生成目标导向视觉导航基准。在超过2,000个场景上，框架生成2,220万条专家轨迹，并公开工具、协议和派生资产。

#### 方法动机分析

3DGS适合大规模真实感渲染，但不直接回答机器人能否站立、移动是否碰撞、目标中心是否可达。传统基准依赖人工网格和标注，3DGS转网格又可能产生损失与修补成本。论文的假设是，高斯密度能提供足够几何证据，多视角分割能提供稳定语义；二者结合即可构造可复现任务。方法目前只覆盖静态场景和平面SE(2)运动。

#### 方法设计详解

输入为冻结3DGS、相机位姿和类别集合。框架依据高斯位置、协方差和不透明度体素化，用高斯重叠计算体素不透明度，超过阈值便判为占用；再以RANSAC估计地面、对齐重力方向，并按机器人高度区间压缩成二维代价图。DBSCAN与AlphaShape找出有重建支持的区域，把缺乏证据的空间也屏蔽，最后按机器人半径膨胀障碍。语义分支从多姿态渲染RGB-D，用SAM3提取掩码，边界腐蚀后反投影到三维；体素投票和高斯中心聚类得到物体实例。系统据此采样起点—目标、做可达与碰撞检查、生成专家轨迹，在统一评估器中支持PointNav、ImageNav和ObjectNav。

#### 方法对比分析

它的贡献是基准构造与评估基础设施，而不是新的导航策略。相较只接入渲染器的方法，NavArena提供可查询的占用代价图；相较人工网格修复，它直接利用密度和支持区域；相较单帧检测，它用多视角融合产生实例目标。因此更适合静态3DGS的大规模、可重复评测，但仍受重建孔洞和伪影影响，不能替代完整动力学仿真。

#### 实验分析（精简版）

0.05 m分辨率下，状态和运动检查的假阴性均为0.000%；CPU查询耗时为0.986 μs和39.65 μs，FCL网格查询则为7.802 μs和425.02 μs。语义有效中心率达92.25%，高于SceneSplat的65.95%和FlashSplat的57.36%。策略示例中，NaviBridger在ImageNav的Distance SR为0.434，Uni-NaVid在ObjectNav为0.374。长程难度会让单任务ImageNav成功率下降约72%–91%，说明闭环误差会累积。局限是只测静态场景、采用SE(2)，且依赖底层重建质量。

#### 实用指南

论文声明将公开生成工具、评估协议和派生资产，但本文未给出具体仓库、依赖版本或训练超参数。推理阶段由冻结3DGS按需渲染RGB-D，评估步数上限为1,500步；论文未说明SAM3推理硬件与神经策略训练超参数。复现需准备3DGS、相机位姿、SAM3、多视角RGB-D、体素分辨率、占用阈值、机器人尺寸及聚类参数，并固定难度划分。换机器人时应重算高度区间、半径膨胀和动作模型；迁移动态环境则必须增加时序障碍处理。

#### 总结

核心思想：让3DGS直接成为导航基准

1. 由高斯密度构造占用与支持区域。
2. 将多视角开放词汇掩码融合成物体目标。
3. 检查可达性并自动生成专家轨迹。
4. 用统一闭环协议评测三类导航任务。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.04602v1)
- [arXiv](https://arxiv.org/abs/2609.04602v1)

---

<a id='2609.04759v1'></a>
## [Dressing in Motion: A Human Motion-Aware Diffusion Policy for Robot-Assisted Dressing](https://arxiv.org/abs/2609.04759v1)

**Authors:** Haoxiang Sun, Fangyuan Wang, Songhao Huang, Justina Y. W. Liu, Jihong Zhu, Peng Zhou, David Navarro-Alarcon

**Published:** 2026-09-04

**Categories:** cs.RO

**Abstract:**

Robotic dressing assistance is a promising solution for supporting older adults with physical impairments in daily living. However, dressing under human motion remains challenging, as complex garment--human contact and occlusions make it difficult to generate actions aligned with arm movements. In this letter, we propose a visuomotor policy that learns dressing skills from static expert demonstrations and generalizes to dynamic user-motion scenarios. A diffusion policy tailored to garment--human interaction geometry learns from partially observed point clouds with varied arm postures. We then introduce an object-centric representation based on PDE diffusion to capture the axial distribution of the arm. By sampling motion-relevant regions and registering them across consecutive observations, the proposed method approximates arm motion and reactively adapts the executed trajectory. We evaluate our method in simulation and a real-world human study involving nine participants, three garment types, and six arm-motion patterns. Results show that our method outperforms baselines in dressing progress, freedom of movement, and user comfort. The project website is https://anonymous.4open.science/w/dressing-in-motion.

### 论文解读

#### 摘要翻译
机器人辅助穿衣有助于老年人和行动受限者，但人在穿衣时会自然移动手臂，衣物形变、人体遮挡和复杂接触让动作难以对齐。本文提出一种视觉运动策略：用静态专家演示训练扩散策略，再从局部点云中提取手臂运动线索，使机器人能适应动态用户。仿真和真实用户实验表明，该方法提升了穿衣进度、运动自由度与舒适度。

#### 方法动机分析
现有方法常把手臂视为静止目标，用户必须保持“木头人”姿势；一旦疲劳、打电话或取物，固定轨迹就会偏离。完整重建衣物和被遮挡手臂既昂贵又不稳定。本文的核心假设是：无需恢复全部人体状态，只要在连续观测中找到与手臂轴向相关的局部区域，就能估计运动并修正动作。

#### 方法设计详解
流程是“局部点云与末端位姿→扩散策略生成动作块→运动区域提取→配准→轨迹修正”。点云先经最远点采样，再由 EdgeConv 编码局部几何，并与机器人状态融合。扩散策略使用 DDIM 从噪声逐步生成穿衣轨迹，训练数据只含静态专家示范。为定位手臂，方法在点云上求解 PDE 扩散标量场，以肩部为源点形成沿手臂平滑变化的对象中心表示；采样运动相关 ROI 后，用 GICP 配准当前与参考 ROI。所得变换将静态轨迹投影到新姿态，并在 SE(3) 中用 Exp/Log 映射限制修正幅度，减少误差导致的突变。扩散策略以 15 Hz 推理、预测步长为 8，自适应模块超过 50 Hz。

#### 方法对比分析
与 DP3、Diff-MPC、BC-LSTM 以及不带自适应模块的版本相比，本文把“学会穿衣”和“跟随人体”分成两个互补层次：生成模型处理衣物—人体交互，几何模块处理运行时位移。它不要求动态动作训练数据，适合隐私敏感、难以大量采集动态示范的辅助穿衣场景；但高速、大幅度动作和初始对齐仍是边界。

#### 实验分析（精简版）
仿真使用 180 条遥操作轨迹，真实研究覆盖 9 名受试者、3 类衣物和 6 种手臂动作。动态仿真中，速度为 1.0 和 2.0 时本文方法 DR 仍高于 0.95；真实实验平均 DR=0.88±0.105、成功率为 89%±10.4%，而 DP3 的 DR 约为 0.5–0.7，Diff-MPC 与 BC-LSTM 在复杂动作下低于 0.3。去掉自适应后动态任务明显退化。DDIM 去噪小于 50 ms、GICP 小于 10 ms，说明该设计具备实时性；不过剧烈初始运动仍可能造成袖口插入失败。

#### 实用指南
复现时应准备 UR10e、RealSense D435i 或等价点云传感器，并按“EdgeConv 编码—DDIM 动作块—PDE 标量场—ROI/GICP—受限 SE(3) 投影”实现。论文报告的训练数据为仿真 180 条、现实 210 条静态演示，部署时需保持 15 Hz 策略和超过 50 Hz 自适应更新。论文给出匿名项目主页，但未明确说明完整代码和权重是否公开；迁移到其他机器人时应重新校准末端动作空间、ROI 阈值和安全限幅。

#### 总结
核心思想：静态技能驱动动态穿衣

1. 采集人体与衣物的局部点云及末端状态。
2. 用静态演示训练扩散策略生成穿衣动作块。
3. 用 PDE 标量场找到手臂相关区域并以 GICP 估计运动。
4. 在 SE(3) 中限幅投影轨迹，实时执行并适应用户。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.04759v1)
- [arXiv](https://arxiv.org/abs/2609.04759v1)

---

<a id='2609.04802v1'></a>
## [Linguistic Trajectory Encoding for Efficient Long-Horizon Spatial Memory in Embodied Agents](https://arxiv.org/abs/2609.04802v1)

**Authors:** Tianyidan Xie, Shenyi Wang, Qiang Tang, Mingjie Wang, Zhicheng Qiu, Xuanfu Li, Zhan Xu, Jian Yang, Lanjun Wang, Zili Yi

**Published:** 2026-09-04

**Categories:** cs.CV, cs.AI

**Abstract:**

Embodied agents performing long-horizon tasks require a memory representation in which the state transitions of dynamic objects remain queryable in natural language across hours-to-days observation horizons. Existing systems either drop fine-grained motion (clip-level video-language embeddings), keep it only as raw coordinates (geometric SLAM), or organise it around immediate task context (agent working memories). None of them gives the agent a per-object timeline whose state transitions are themselves queryable in language. Our key contribution is \textbf{Linguistic Trajectory Encoding} (LTE), which compresses dynamic object motion histories via a hybrid representation combining natural language descriptions, sparse spatial anchors, and visual anchors. LTE adapts compression to motion complexity by anchoring periods without reliable observations to the last seen location, while representing motion with geometric waypoints and linguistic descriptions to preserve accuracy. To evaluate these capabilities across extended time horizons, we construct the \textbf{Spatial Memory Benchmark} (SMB) from EgoLife multi-day recordings, targeting capabilities absent in existing benchmarks: semantic trajectory retrieval and long-horizon object retrieval. On SMB, the LTE-based system achieves $45.3\%$ success in semantic trajectory retrieval and $48.7\%$ in long-horizon object retrieval, outperforming structured-memory and VLM baselines (best prior: $31.9\%$ and $34.4\%$). LTE achieves trajectory compression by factors of $8.7\times$ to $26.1\times$ with sub-second query latency on $24$\,h video. On Ego4D natural-language queries, the system reaches $28.75\%$ / $55.10\%$ R@1/R@5, $+15.80$ / $+31.30$ pts over EgoVLPv2.

### 论文解读

#### 摘要翻译

长时段具身任务需要一种记忆表示，使动态物体的状态转移在数小时至数天后仍可用自然语言查询。现有方法要么丢失细粒度运动，要么只保存原始坐标，要么围绕即时任务组织记忆。论文提出语言轨迹编码（LTE），以自然语言描述、稀疏空间锚点和视觉锚点混合压缩对象运动历史，并对复杂度自适应。基于 EgoLife 构建空间记忆基准 SMB 后，系统在语义轨迹检索和长时对象检索上达到 45.3% 和 48.7%，优于最佳先前结果 31.9% 和 34.4%；轨迹压缩为 8.7×–26.1×，24 小时视频查询低于 1 秒。

#### 方法动机分析

具身智能体要回答“洗过的苹果最后在哪里”，必须同时理解对象、状态、时间和位置。视频语言嵌入没有稳定的三维索引，几何 SLAM 没有“洗过”等语义，任务记忆又会忽略非当前任务对象。LTE 的核心假设是运动历史具有语义冗余：语言能概括阶段，几何提供位置，视觉确认身份。系统也明确边界：单摄像头看不到视野外的移动，只能报告最后已知状态并标注置信度。

#### 方法设计详解

输入连续 RGB 视频与音频，SAM3 跟踪对象，ViPE 重建三维位置，Qwen3-VL-8B-Instruct 生成房间标签和运动字幕，Whisper 与 Qwen3-8B 提取语音事件。每个对象保存三部分：带时间区间的运动字幕、稀疏三维锚点、锚点视觉裁剪。跟踪中断达到 2 秒时，将区间锚定在最后位置；连续运动则用 Douglas–Peucker 以 0.15 m 容差保留转折点，并在关键点和状态边界保存视觉证据。静态环境用八叉树做区域剪枝，五个互联视图提供场景、对象、文本、事件和图像访问。查询先解析对象、时间、区域、语义，再路由、过滤、聚合；STR 用文本相似度匹配字幕，LOR 返回时间窗内最近锚点，视觉查询匹配视觉锚点。稳定场景从 3 秒采样步长退避到最多 15 秒。系统在单张 A800 80 GB 上运行，24 小时在线查询延迟为 0.43 秒。

#### 方法对比分析

LTE 位于几何 SLAM 与视频语言模型之间：不是保存密集坐标，也不是只匹配片段，而是以“每个对象的语言化运动阶段”为中心，并让每个阶段连接空间和视觉证据。相比关键帧记忆，它保留连续状态变化；相比事件记忆，它不把对象历史折叠进宽泛活动。适合需要长期追踪对象状态、位置和身份的家庭机器人或空间助手，但依赖稳定跟踪和三维重建。

#### 实验分析（精简版）

SMB 来自 EgoLife 的 300 小时、6 人、7 天记录，共 600 个查询。LTE 的 STR/LOR 为 45.3%/48.7%，相对 Qwen3-VL-235B+Grounding-DINO 的 31.9%/34.4% 提升 13.4/14.3 个百分点；在 IoU≥0.5 时仍提升 12.4/13.5 个百分点。24 小时查询为 0.43 秒，而基线为 98.3 秒；内存压缩达到 26.1×。消融中去掉文本字幕使 STR 降至 33.5%，说明语言化状态是关键贡献。主要风险来自跟踪 ID 切换、字幕歧义和单目空间漂移。

#### 实用指南

论文给出了可复现的模型组合、阈值和硬件，但未提供独立代码仓库或模型下载链接，也未报告端到端微调超参数。复现时需实现 SAM3/ViPE 感知、0.15 m 轨迹简化、2 秒丢失阈值、八叉树索引及 0.8 的文本匹配阈值。迁移到其他机器人或场景，应替换检测跟踪器、坐标系和房间类别，并重新生成对象字幕与锚点；跨办公室、仓库和户外的泛化仍需验证。

#### 总结

核心思想：语言化对象轨迹

1. 跟踪对象并重建三维运动。
2. 用字幕概括运动阶段。
3. 以稀疏空间点和视觉裁剪绑定语义。
4. 通过八叉树与多视图快速检索。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.04802v1)
- [arXiv](https://arxiv.org/abs/2609.04802v1)

---

<a id='2609.05325v1'></a>
## [FIRE-LIVWO: Robust LiDAR-Inertial-Visual-Wheel Odometry via Failure-Immune mmWave Radar Enhancement](https://arxiv.org/abs/2609.05325v1)

**Authors:** Kun Hu, Menggang Li, Kaidi Wu, Zhiwen Jin, Yingjie Zhao, Chaoquan Tang, Eryi Hu, Gongbo Zhou

**Published:** 2026-09-04

**Categories:** cs.RO

**Abstract:**

Achieving robust SLAM in large-scale underground coal mines with complex structures and severe degeneracies remains highly challenging. Dense smoke and dust cause substantial loss of visual information and degrade LiDAR point-cloud features, while long, self-similar corridors induce geometric degeneration, leading to pronounced odometry drift. To address these issues, we propose FIRE-LIVWO: Failure-Immune mmWave Radar-Enhanced LiDAR-Inertial-Visual-Wheel Odometry, a tightly coupled multi-modal odometry framework based on an iterated error-state Kalman filter (IESKF). The framework fuses 4D mmWave radar, LiDAR, and visual features within a unified VoxelMap and jointly constructs LiDAR-radar point-to-plane residuals and sparse visual photometric residuals. In smoke-filled environments, we exploit the strong penetration of 4D mmWave radar and introduce pointwise Doppler velocity constraints to preserve state observability. In geometrically degenerate corridors, we tightly couple wheel odometry using non-holonomic constraints (NHC) and online lever-arm compensation to reduce drift. Our central contribution is a degeneration detection and adaptive fusion model switching strategy grounded in geometric and visual observability analysis, which quantifies observability online and dynamically adjusts modality weights. Real-world experiments in underground coal mines demonstrate that FIRE-LIVWO accurately identifies failure boundaries, enabling reliable modality switching under extreme conditions. Compared with baselines, it achieves superior accuracy and robustness (average localization error of 5.677m). We open source our code on Github to benefit the robotics community.

### 论文解读

#### 摘要翻译

地下煤矿的浓烟、粉尘和低照度会破坏视觉与LiDAR特征，长而相似的巷道又造成几何退化和里程计漂移。论文提出FIRE-LIVWO：基于迭代误差状态卡尔曼滤波的紧耦合LiDAR、IMU、视觉、轮速与4D毫米波雷达系统。它在统一VoxelMap中联合LiDAR/雷达点面残差和稀疏视觉光度残差；烟雾中利用雷达穿透性及逐点Doppler速度约束维持可观测性；几何退化时以非完整性约束和在线杆臂补偿融合轮速。系统依据视觉与几何可观测性在线切换模态，真实矿井实验平均定位误差为5.677 m。

#### 方法动机分析

固定权重融合无法应对“视觉失效”和“结构欠约束”交替出现的矿井环境，而稀疏雷达也不宜直接替代高精度LiDAR。论文的核心假设是：不同退化模式仍会留下互补证据——烟尘不一定破坏雷达Doppler，长走廊不一定破坏车辆本体运动。因而应先检测观测的可用程度，再把合适约束加入统一估计器。

#### 方法设计详解

IMU前向传播提供先验，LiDAR反向传播补偿扫描运动；LiDAR、雷达候选点进入根体素0.5 m的统一八叉树VoxelMap。IESKF联合优化LiDAR/雷达点面、雷达径向速度、稀疏直接视觉和轮速残差。Dark Channel Prior估计平均透射率作为视觉可观测性；点面残差Jacobian形成的Hessian特征值则衡量几何在平移、旋转方向上的欠约束。正常时运行LIV，视觉失效切至LIVR，几何退化切至LIVW，双重退化启用LIVRW。轮速模块加入车辆非完整性约束与在线杆臂补偿，最终输出六自由度位姿和着色地图。论文给出的硬件为Intel i7 CPU与NVIDIA 1050Ti GPU。

#### 方法对比分析

与只依赖LVI或将雷达作为独立替代的方案相比，FIRE-LIVWO把雷达的几何、Doppler信息和车辆运动学放入同一IESKF，并以观测性驱动激活，而非固定融合。它适合烟尘、重复结构和GPS拒止场景；但收益依赖至少一个补偿模态仍可靠，轮胎打滑或多传感器同时失效时边界尚未验证。

#### 实验分析（精简版）

三个真实煤矿场景覆盖重烟、稀疏特征和重复走廊。完整系统将平均定位误差降低至5.677 m，优于GaRLIO的17.212 m、R3LIVE的31.596 m和4DRadarSLAM的45.453 m；仅LVI的FIRE-Base为23.962 m。FAST-LIVO2及FIRE-LIVW在烟雾中崩溃；移除雷达或轮速后误差升至约15–23 m，说明两类约束分别针对视觉和几何退化发挥作用。实验仍主要是特定矿井验证，未证明所有平台和极端失效条件下均稳定。

#### 实用指南

论文已在GitHub开源代码。复现重点是五类传感器时间同步、外参、0.5 m地图体素、IESKF残差实现，以及透射率和Hessian特征值阈值。迁移到隧道巡检或救援车时需重做标定、雷达坐标模型、车辆运动学和退化阈值；非轮式机器人应以自身可观测的本体约束替换NHC。论文未给出完整学习超参数，因为核心流程是在线滤波与几何估计。

#### 总结

核心思想：按可观测性切换互补传感器

1. 多传感器进入统一地图与IESKF。
2. 透射率检测视觉失效，Hessian检测几何退化。
3. 烟雾启用雷达Doppler，走廊启用轮速NHC。
4. 按退化组合切换LIV、LIVR、LIVW或LIVRW并输出位姿。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.05325v1)
- [arXiv](https://arxiv.org/abs/2609.05325v1)

---

<a id='2609.05300v1'></a>
## [Human-Human & Human-Robot Interaction Transformer (H2INT) for Robot Navigation in Dense and Uncertain Crowds](https://arxiv.org/abs/2609.05300v1)

**Authors:** Ao Shen, Kaixi Chen, Shiwei Liu, Fang Deng, Chen Chen

**Published:** 2026-09-04

**Categories:** cs.RO

**Abstract:**

Safe robot navigation in dense crowds requires reasoning about pedestrian motion and how it may change in response to a robot. However, many learning-based approaches generate pedestrian motion independently of the robot or assume uniform reciprocity, omitting an important source of interaction uncertainty. This paper presents a Human-Human & Human-Robot Interaction Transformer (H2INT), a reinforcement learning framework that retains robot-conditioned changes in pedestrian motion during policy learning while allowing responsiveness to vary across pedestrians. Responsiveness affects the crowd dynamics when the robot is visible but is not supplied as a policy input; the policy must instead infer its consequences from robot-centered relative positions. A two-stage gated Transformer progressively encodes human-human and human-robot relations, while a recurrent policy captures their temporal evolution. A curriculum gradually reduces pedestrian responsiveness to increase interaction difficulty. Simulation experiments demonstrate improved navigation safety and robustness over representative baselines across response conditions and crowd densities, and show transfer without retraining to structurally distinct crowd-flow layouts. Ablations support the hierarchical relational encoding and gated updates. Real-robot deployment further verifies that the learned policy can operate with sparse observations in a physical environment.

### 论文解读

#### 摘要翻译
H²INT 面向稠密且不确定人群中的安全导航。它不把所有行人视为独立障碍，也不假设人人都会以同样方式让行，而是在环境动力学中保留“机器人会改变行人运动”的反馈，并让策略从相对位置中间接推断这种变化。两阶段门控 Transformer 编码交互关系，循环策略保留时间信息，课程学习逐步降低行人响应性。仿真和真实机器人试验显示，该方法兼顾安全性、密度鲁棒性和布局迁移。

#### 方法动机分析
机器人在共享通道中移动时，有人会减速或绕行，有人却不响应；若训练环境把行人轨迹与机器人隔离，策略即使预测能力很强，也学不到真实反馈。这正是论文要解决的研究痛点与动机。论文不试图估计人的认知注意力，而是为每名行人设置 episode 级、彼此不同且随时间保持的潜在响应倾向：机器人位于其视野和感知范围内时，行人可能把机器人纳入 ORCA 交互，也可能忽略它。策略只能观察相对位置历史，因此问题更接近部分可观测的行为推断。

#### 方法设计详解
输入包括机器人 7 维状态、机器人速度和每名行人的 2D 相对位移，不使用行人速度、注视方向或响应标签。机器人与行人分别经 MLP 映射为 token，动态 mask 和零填充支持可变人数，并加入类型/位置编码。两个串联 Transformer 阶段都采用“多头自注意力/FFN + GRU 风格门控残差”：第一阶段建立交互表征，第二阶段在其上进行关系精炼，再取机器人 token 作为全局上下文。该上下文进入 GRU，Actor 输出高斯动作，Critic 估计价值，以 PPO 和 GAE 端到端训练。奖励兼顾到达、碰撞、人与人距离和效率：到达奖励 +10、碰撞惩罚 −20，每步另有 −0.025 时间惩罚。训练从较高响应率开始，成功率稳定后按固定步长降低响应率，逐步暴露于更困难的非合作人群。训练设定还包括 10–30 人密度范围和 Circle Crossing 训练布局；在线推理直接从当前观测输出动作，不展开行人轨迹。

#### 方法对比分析
ORCA、Social Force 等规则法高效但通常依赖对称交互；CADRL、SARL、DS-RNN、AIG-GST 等学习方法则未必在环境中显式保留机器人引起的行人变化。H²INT 的重点不是宣称自注意力新颖，而是把行为反馈、潜在异质响应、两阶段关系精炼和门控更新组合成一条闭环。相比预测再规划，它不展开长时轨迹，直接从当前机器人中心观测输出动作，适合有稀疏几何行人观测的移动机器人；但仍受仿真响应模型和最大 token 数限制。

#### 实验分析（精简版）
每项评测使用 500 个 episode。在 20 人场景中，响应率从 High 的 0.7 降到 Low 的 0.3，H²INT 成功率仍为 99% 到 94%，碰撞率为 1% 到 6%；Low 下 AIG-GST 为 82% 成功率、18% 碰撞率。完全不响应且人数增至 30 时，H²INT 成功率 88%、碰撞率 12%，AIG-GST 分别为 70% 和 28%，成功导航时间为 16.46 秒。只在 Circle Crossing 训练后，迁移到三种结构不同的群流布局，30 人下 High/Medium/Low 成功率仍达 99%/93%/91%。但真实验证是室内单次无碰撞运行，论文明确指出还需要多场地、不同密度的定量现场研究。

#### 实用指南
复现需要改进的 CrowdNav/ORCA 环境、PPO+GAE、响应率退火、FOV 与感知范围、响应状态去抖、padding mask，以及统一的成功/碰撞/超时评测。真实部署使用 2D LiDAR 配合 DR-SPAAM 提供相对行人位置，动作转换后经 ROS cmd_vel 发布。论文未说明代码和模型权重是否开源，也未给出完整训练超参数，因此不能把它视为开箱即用方案。迁移到新机器人时需重新适配尺寸、速度限制、动作接口和真实人群反馈；若保持相对几何输入，方法可作为稀疏感知导航策略的起点。

#### 总结
核心思想：让导航策略学习人群反馈

速记 pipeline：
1. 观察机器人状态与相对行人位置。
2. 两阶段门控 Transformer 精炼人-人/人-机关系。
3. GRU 记忆交互变化，Actor-Critic 输出动作。
4. 逐步降低行人响应率，检验密集与未见群流。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.05300v1)
- [arXiv](https://arxiv.org/abs/2609.05300v1)

---

<a id='2609.05260v1'></a>
## [One Word, Different Action: A Real-Robot Benchmark for Language-Conditioned Embodied Reasoning](https://arxiv.org/abs/2609.05260v1)

**Authors:** Yiwei Liu, Luwei Yang, Shunbo Lei

**Published:** 2026-09-04

**Categories:** cs.RO

**Abstract:**

Natural-language instruction changes can directly alter robot behavior. A reliable embodied system should preserve its action when the task is unchanged and update it correctly when the task itself changes. We introduce One Word, Different Action, a real-robot benchmark built on physical decision states and executable actions, using task-preserving and task-changing instruction pairs to jointly evaluate Decision Invariance and Decision Sensitivity, with further evaluation under multi-constraint reasoning and real-RGB grounding. Experiments show that modern models are near saturation on single-constraint instruction changes, yet several models degrade noticeably when multiple task constraints must be integrated into one executable decision. These results suggest that the more salient remaining challenge is no longer recognizing an isolated instruction change, but reliably composing multiple task requirements into a correct robot action decision.

### 论文解读

#### 摘要翻译

论文提出 OWDA（One Word, Different Action）真机基准：在同一物理决策状态和候选动作空间中，构造“任务不变但换说法”和“任务约束真的改变”的指令对，分别考查机器人是否保持动作、是否正确更新动作，并扩展到多约束推理和真实 RGB 输入。结果显示，单约束变化已接近饱和，多个约束合并成一个可执行决策仍会造成模型依赖的退化。

#### 方法动机分析

仅看任务成功率，无法区分“同义改写导致不必要换动作”和“任务变了却沿用旧动作”。OWDA的核心假设是固定物理状态后，动作变化应只由任务约束变化触发，而非由语言表面变化触发；因此它把语言理解直接连接到可执行、可机器核验的离散决策。

#### 方法设计详解

流程是：固定真实机器人状态及候选物体/动作，定义确定性约束并计算唯一正确动作，再生成多种自然语言指令。任务覆盖动作抑制、集合保留/排除、选择数量和执行顺序；复合任务要求动作集合同时满足多个条件。模型输入为结构化状态或真实 RGB 加最小候选元数据，输出单动作、动作集合或有序序列，必须精确匹配目标。指标将配对成员同时纳入：任务不变时测 Decision Invariance，任务改变时测 Decision Sensitivity，并统计过度敏感、过度不变和解析失败。基准含85个物理锚点、988个原子任务族、3952个实例，另有84个多约束实例；RGB子集为748个实例、187个任务族。按物理锚点切分避免同一状态泄漏。评测温度为0、最大输出512 tokens；Qwen3.5-27B关闭 reasoning。

#### 方法对比分析

与只测独立指令成功率或单纯同义改写鲁棒性的基准相比，OWDA同时测“该不该保持”和“该不该改变”，并在相同物理基底上构造反事实对。它的贡献是诊断框架、确定性动作表示和多约束组合测试，而不是新的控制策略。适合分析语言模型的动作决策可靠性，但不等同于低层轨迹控制或完整机器人安全评估。

#### 实验分析（精简版）

单约束文本评测中，Qwen3.5-27B精确准确率达到0.994，Gemma4-31B达到1.000。多约束时Qwen从原子0.997降至复合0.964，GLM-5-FP8从0.987降至0.940；Gemma仍为1.000，说明瓶颈在约束整合而非识别单个变化。结构化状态改为RGB后，Gemma精确准确率仅由1.000变为0.997。遮罩任务关键操作词后，Qwen准确率降至0.558，支持其确实依赖决定性语言。局限是离散、受控的动作空间，未覆盖连续控制和复杂执行扰动。

#### 实用指南

复现要按物理锚点而非句子切分；严格解析集合成员和时序，解析失败仍计入分母，并用任务族 bootstrap 计算置信区间。论文未说明代码、模型或数据已开源，不能假定可直接下载。迁移时需重建本机器人可验证的候选动作、物理锚点、语言约束和安全检查；若要评价真实执行，还应增加轨迹误差、碰撞和时延指标。

#### 总结

核心思想：只在任务变时改变动作

1. 固定状态与候选动作。
2. 同义改写检查动作保持。
3. 约束变更检查动作更新。
4. 叠加数量和排除条件，测试组合推理。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.05260v1)
- [arXiv](https://arxiv.org/abs/2609.05260v1)

---

<a id='2609.04718v1'></a>
## [HiSfM: Disambiguating Structure-from-Motion via Scaffold-Anchored Hierarchical Reconstruction](https://arxiv.org/abs/2609.04718v1)

**Authors:** Ziding Zhao, Hainan Cui, Peilin Tao, Shuhan Shen

**Published:** 2026-09-04

**Categories:** cs.CV

**Abstract:**

Structure-from-Motion (SfM) is a fundamental tool for sparse 3D reconstruction with broad impact in robotics and vision, supporting mapping, localization, and large-scale scene modeling. However, conventional pipelines often fail under hard visual ambiguity caused by repeated or symmetric structures, and incur heavy computational cost due to redundant cameras and constraints. We present HiSfM, a hierarchical coarse-to-fine SfM framework that improves robustness and efficiency through scaffold construction. HiSfM first forms strong local communities using geometrical induced heuristics, then connects communities with a compact yet strong skeleton by packing edge-disjoint spanning trees (EDST) while verifying skeletal edges with a two-view disambiguator. We reconstruct a stable scaffold on this verified skeleton, serving as an anchor to capture the essence of the scene, and subsequently absorb remaining images via efficient registration and triangulation for further refinements. Experiments on ambiguity-focused benchmarks and general datasets show that HiSfM prevents ambiguity-induced failures while substantially reducing runtime compared to previous methods, and improves completeness over aggressive sparsification methods. Code is available at https://github.com/3dv-casia/HiSfM.

### 论文解读

#### 摘要翻译
结构光束法平差（SfM）是稀疏三维重建的基础工具，服务于机器人建图、定位和大规模场景建模。但重复或对称结构造成的视觉歧义会使传统流程失败，冗余相机与约束又带来很高计算成本。论文提出 HiSfM，一种通过脚手架构建提升鲁棒性和效率的层级化粗到细 SfM 框架：先用几何启发式形成强局部社区，再以边不相交生成树连接社区，并用双视图去歧义器验证骨架边；随后重建稳定脚手架，作为吸收剩余图像、注册和三角化的锚点。实验表明它能避免歧义诱发的失败、降低运行时间，并比激进稀疏化方法获得更高完整度；代码已开源。

#### 方法动机分析
重复建筑、对称纹理中，错误匹配可能通过常规几何检验，产生看似合理却重叠或幻影的模型。密集采集还会制造近重复图像与过多约束，使增量 SfM 和束调整难以扩展。逐边使用 Doppelgangers++ 虽稳健却昂贵，CamTrip 等激进稀疏化虽快又可能切碎场景。HiSfM 的核心假设是：社区内部强几何关系较可靠，风险主要集中在连接不同社区的桥接边。

#### 方法设计详解
输入是以图像为节点、以内点数为边权的视图图。首先保留每个图像的最强邻居（top-1），得到局部社区；再在社区层面按权重贪心挑选桥接边，打包成 K 棵边不相交生成树（EDST），最大化保留的几何支持，并只对这些骨架边调用 DG++ 去歧义。K 按图像规模取 max(1, round(n/300))，多棵树提供替代连接，避免单桥失败导致分裂。叶节点通过社区内高权重边接回。然后在验证骨架上用 COLMAP 增量 SfM 建立脚手架，利用其三维点对剩余图像做 PnP 注册，迭代三角化和局部 BA，最后全局 BA 输出完整模型。

#### 方法对比分析
HiSfM 的关键区别是“关键桥边去歧义、骨架保连通、剩余图像锚定注册”的组合。DG++ 把鲁棒验证扩展到大量边，成本高；CamTrip 更强调稀疏和速度，可能过分割；HiSfM 通过 EDST 保留 K-边连通性，再把昂贵验证限制在全局风险最高的边上。因此它更适合大规模、重复或对称结构场景，但若社区内部本身存在系统性歧义，其稳定性假设会成为边界。

#### 实验分析（精简版）
论文在视觉歧义压力测试集、1DSfM 和 Photo Tourism 上比较 COLMAP、CamTrip 与 DG++。Trafalgar 中 HiSfM 用时 170.4 分钟，DG++ 为 3035.3 分钟，约快 17 倍且完整度更高；Berliner Dom 的 1606 张图像重建仅需 17.9 分钟，DG++ 需 598.8 分钟，重投影误差为 0.8 像素。结果支持效率、连通性和抗歧义性的折中，但收益依赖数据集、匹配质量和硬件。

#### 实用指南
代码链接为 GitHub 的 3dv-casia/HiSfM。复现应保留内点数边权，采用 top-1 社区、EDST 骨架和仅验证桥边的策略，并按骨架 SfM、PnP、三角化、局部/全局 BA 顺序执行。实验使用 Intel i7-14700K、RTX 3090（24 GB）、128 GB RAM 和 COLMAP；论文未完整说明全部依赖与运行参数。迁移到机器人数据时需重新检查社区歧义、调整 K 与匹配阈值，标准 PnP 和 BA 可复用。

#### 总结
核心思想：骨架锚定，关键边去歧义

1. 强邻居聚成局部社区。
2. EDST 构建多重连通骨架并验证桥边。
3. 骨架先做增量 SfM，固定场景锚点。
4. 剩余图像以 PnP、三角化和 BA 高效吸收。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.04718v1)
- [arXiv](https://arxiv.org/abs/2609.04718v1)

---

<a id='2609.05178v1'></a>
## [LIBERO-RECOVER: Beyond Task Success Towards Failure Recovery in Robotic Manipulation Models](https://arxiv.org/abs/2609.05178v1)

**Authors:** Lin Liu, Zhicheng Bao, Lu Zhang, Ziying Song, Wu Yang, Shuai Tao, Wulong Liu, Huchuan Lu

**Published:** 2026-09-04

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) or World Action (WAM) models have recently demonstrated remarkable performance in robotic manipulation. On LIBERO, SOTA method have achieved nearly 100\% success rates, seemingly suggesting that the models are ready for deployment in real world. However, near perfect performance on existing benchmarks can be misleading: success under ideal conditions does not imply real world robustness. Existing benchmarks primarily evaluate task completion from predefined initial states, while real world interactions inevitably involve failures such as failed grasps, collisions, and unintended object movements. A robot must therefore not only execute tasks successfully, but also recognize and recover from failures to continue the task. Yet this capability remains largely unmeasured, revealing a critical gap between benchmark performance and real world reliability. To address this gap, we introduce LIBERO-Recover Benchmark, a large scale benchmark for failure recovery in robotic manipulation. Built upon LIBERO, we collect real execution failures from SOTA embodied models and construct 1,000+ scenarios across four recovery levels: (1) Action Retry, (2) Action Adaptation, (3) Object State Recovery, and (4) Environmental Recovery. We evaluate four core capabilities: spatial understanding, object structure reasoning, interaction understanding, and topological reasoning. As the first large-scale benchmark for embodied failure recovery, LIBERO-Recover shifts evaluation from \emph{Can the robot succeed?''} to \emph{Can the robot recover after failure?''}, promoting robust and generalizable embodied agents. The project will be avaible in \textcolor{blue}{https://liulin815.github.io/LIBERO-Recovery/}.

### 论文解读

#### 摘要翻译

VLA/WAM 模型在 LIBERO 上接近满分，但理想初态下的成功不等于现实鲁棒性。论文提出 LIBERO-Recover，从具身模型执行中收集自然失败，构建覆盖动作重试、动作适配、物体状态恢复和环境恢复四级的规模化基准，把问题从“能否成功”改成“失败后能否恢复”。

#### 方法动机分析

传统基准多只评估从预定义初态一次完成任务，忽略抓取滑脱、碰撞和物体意外移动造成的后果。论文的核心假设是：可靠机器人必须识别失败、理解状态变化、重新规划并续接原任务；手工改初始姿态只能测分布泛化，不能充分测执行中失败的因果恢复。

#### 方法设计详解

流程输入是 LIBERO 指令、初始场景及多种策略的执行视频和模拟器状态。首先在 4 个任务套件、130 个子任务上运行策略，不注入扰动，得到失败轨迹；再由 Qwen3.5-27B-Instruct 定位失败前、中、后三段，从时间边界回取失败状态与物体位姿；最后按后果标注四级恢复场景。L1 只需重试，L2 需根据观察调整动作，L3 先恢复任务相关物体，L4 还要处理阻塞任务的环境拓扑。每个场景包含指令、初态、失败轨迹、失败状态、目标和恢复行为。评估使用恢复成功率 RSR、退化率 RD 和跨失败状态一致性 RC；4 名操作员另采集 3,184 条轨迹用于微调。

#### 方法对比分析

它区别于手工改变物体位姿的鲁棒性测试：场景来自策略真实执行造成的状态转移。四级难度把局部动作修正与物体/环境状态恢复分开，能揭示模型由“会调下一步”到“理解状态并重建计划”的能力断层。WAM 的动作条件状态建模可能更利于恢复，但论文将其视为结果解释而非因果证明。

#### 实验分析（精简版）

论文评估 OpenVLA-OFT、π0-Fast、GR00T-N1.5、π0、Wan2-Policy、Cosmos-Predict2-Policy，每任务 10 次。跨模型平均恢复后成功率在 LIBERO-100 从 26.2% 降至 0.3%，RD=100%；Spatial 从 15.0% 降至 6.7%，RD=54%。Wan2-Policy 标准 LIBERO-100 比 GR00T 高 14.40 个百分点，却在失败恢复低 5.0 个百分点。L1/L2 明显优于 L3/L4；较小 action chunk 和初始帧时间上下文更有利。恢复微调提升恢复集，却对普通 LIBERO 迁移有限，甚至略降。

#### 实用指南

项目主页已给出，但完整代码、权重和数据的发布状态需以实际页面核对。复现需保持官方配置、每任务 10 次、轻微位置扰动和 1.1 倍人类时限，并保存视频与模拟器状态；再完成失败定位、L1–L4 标注及 RSR/RD/RC 评估，同时审计自动标注质量。迁移到新机器人或数据集时要重做状态回取、可恢复性标签和恢复数据；论文未报告统一学习率、batch size、epoch、硬件或延迟。

#### 总结

核心思想：让机器人学会失败后自救

1. 运行策略收集自然失败。
2. 定位偏离并重建失败状态。
3. 按四级难度执行恢复。
4. 用 RSR、RD、RC 检验闭环自救。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.05178v1)
- [arXiv](https://arxiv.org/abs/2609.05178v1)

---

<a id='2609.05324v1'></a>
## [RoboSPA: Can VLA Models Go Beyond Simple Scenes and Short-Horizon Tasks?](https://arxiv.org/abs/2609.05324v1)

**Authors:** Zhenxuan Fan, Bo Zhang, Yutong Lin, Yuqian Yuan, Juekai Lin, Liang Liang, Zhuoyi Huang, Wenqiao Zhang, Juncheng Li, Siliang Tang, Jun Xiao, Yueting Zhuang

**Published:** 2026-09-04

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Vision-Language-Action (VLA) models have shown promising progress in language-conditioned robotic manipulation. However, existing datasets and benchmarks mainly evaluate task completion under predefined settings, offering limited insight into model reasoning under increasing spatial and procedural complexity. We introduce \textbf{RoboSPA} (\textbf{Robo}t \textbf{S}patial-\textbf{P}rocedural \textbf{A}ssessment), a large-scale robotic manipulation dataset and benchmark for diagnosing embodied reasoning in VLA models. \texttt{RoboSPA} focuses on two core dimensions, Fine-Grained Spatial Reasoning and Long-Horizon Procedural Planning, covering 10 task categories and 56 base tasks. Each task is instantiated across five difficulty levels, yielding 280 variants with increasing spatial ambiguity and procedural complexity. We collect 527K trajectories across multiple embodiments and diverse scenes. Beyond binary success rate, \texttt{RoboSPA} introduces diagnostic metrics for more detailed evaluation. Experiments on representative VLA models show that current systems still struggle with complex spatial relations, precise low-level execution, and memory-intensive planning. These results establish \texttt{RoboSPA} as a challenging diagnostic benchmark for developing more capable, reliable, and generalizable embodied agents. Our data and code are available at https://github.com/fanzhenxuan/RoboSPA.

### 论文解读

#### 摘要翻译

现有视觉—语言—动作（VLA）基准多考察预设场景中的短流程完成率，难以揭示模型面对空间歧义和长程序时是否真正具备具身推理。RoboSPA提出一套空间—程序化评测：覆盖10类能力、56个基础任务和5级难度，形成280个变体，并收集527K条多具身轨迹。实验表明，当前VLA在复杂空间关系、精细低层执行和记忆密集规划上仍明显薄弱。

#### 方法动机分析

只看成功率会把“找错目标”“抓取失败”“忘记前序信息”混为一谈，也无法观察复杂度上升后的退化曲线。RoboSPA的驱动力是把空间消歧与长时域程序作为独立压力轴，核心假设是：目标接地、动作执行、顺序控制和记忆应被分别诊断，才能指导模型改进。

#### 方法设计详解

输入是RGB观察和自然语言指令，VLA在SAPIEN/RoboTwin 2.0环境中逐步输出机器人动作。空间推理包含几何属性、距离、规范位置、指称关系和跨视角五类；程序规划包含重复、无序、有序、复合协调和记忆密集五类。每个基础任务沿空间歧义或程序复杂度构造L1–L5，共280个变体。除成功率外，ONTA=((SR−1/n)/(1−1/n))×100，用于扣除n个候选物体下的随机选择机会；PS=(1/N)Σ(c_i/T)×100，用于统计每个episode完成的子任务比例。实验比较RDT、GO-1、π0.5和X-VLA；每难度使用50条轨迹，微调20,000步、batch size 16，每变体100次rollout。

#### 方法对比分析

RoboSPA不是新的控制模型，而是把复杂度可控化、把二元成功细化为目标准确性和过程进度。与只测简单短任务的基准相比，它能区分“理解了但没抓住”和“根本找错对象”，也能揭示长流程的顺序与记忆瓶颈。它适合评估VLA的可靠性和可扩展性，但当前仍以桌面仿真为主，真实机器人迁移边界尚未验证。

#### 实验分析（精简版）

整体平均SR在L1/L5分别为：RDT 16.8%/6.9%，GO-1 25.1%/8.8%，π0.5 55.2%/22.3%，X-VLA 50.4%/19.9%。X-VLA空间推理最佳，平均SR为41.9%/23.9%，但最难级别平均ONTA仅10.8；π0.5程序规划最佳，平均SR为71.2%/23.1%，PS也从L1的71.2降至L5的45.4%。记忆密集规划在L5对四个模型的SR均为0%，说明长时域记忆和精确执行是关键短板。空间任务中错误目标接地占失败的70%以上，程序任务约一半失败来自低层操作错误，难度提升会同时放大感知、控制和记忆问题。

#### 实用指南

代码和数据已在RoboSPA项目仓库发布。复现时应固定具身、场景随机化和难度，并同时报告SR、ONTA、PS；每难度50条轨迹、20,000步微调、batch size 16和每变体100次rollout是重要设定。迁移到真实机器人需重做动作映射、相机与物体标定及抓取判据；若要定位瓶颈，可针对空间指称、顺序执行和外部记忆分别做消融。

#### 总结

核心思想：分解复杂度，诊断VLA推理

1. 生成多具身、多场景、五级难度任务。
2. 输入视觉与指令，执行VLA动作轨迹。
3. 用ONTA检查找对目标，用PS检查过程完成度。
4. 按空间、执行、顺序和记忆错误定位能力短板。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.05324v1)
- [arXiv](https://arxiv.org/abs/2609.05324v1)

---

