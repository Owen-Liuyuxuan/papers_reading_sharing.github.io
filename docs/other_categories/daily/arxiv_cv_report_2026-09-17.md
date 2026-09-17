time: 20260917

# Arxiv Computer Vision Papers - 2026-09-17

## Table of Contents

1. [PASSAGE: Scaling Scene-Aligned Motion Learning for Perceptive Humanoid Traversal in Cluttered Environments](#2609.18732v1)
2. [PointZero: 3D Point Track Completion for Learning Transferable 3D Dynamics](#2609.19142v1)
3. [ActiveScale: Scaling Active Perception for Robots across Model, Data, and Hardware](#2609.18514v1)
4. [RecMorph: Topology-Guided Spatial Recurrence for Generalized Morphology Control](#2609.18359v1)
5. [Accuracy- and Real-Time-Aware 4D Radar Preprocessing for Autonomous Driving Perception Systems](#2609.18542v1)
6. [UAVs Meet Embodied Intelligence: Bridging Human Intents and Flying Dynamics Via Harnessing Physical-Digital AI Agents](#2609.18326v1)
7. [InterMASH: A Unified Geometric Representation for Grasp Synthesis](#2609.18504v1)
8. [Learning Holistic Whole-Body Loco-Manipulation with a Bipedal Mobile Manipulator](#2609.18930v1)
9. [In-Context Robot Learning with VLM Agents](#2609.19138v1)
10. [CaSCo: Cascade-Aware Soft-Collision Motion Planning](#2609.18910v1)

---

## Papers

<a id='2609.18732v1'></a>
## [PASSAGE: Scaling Scene-Aligned Motion Learning for Perceptive Humanoid Traversal in Cluttered Environments](https://arxiv.org/abs/2609.18732v1)

**Authors:** Yuxuan Ma, Zicheng Zeng, Chunlin Peng, Zhoujian Li, Zetong Zhao, Zhikai Zhang, Yunrui Lian, Han Xue, Sikai Liang, Weiyi Zhu, Mulin Chen, Chenghuai Lin, Jiayu Zeng, Yanwei An, Songan Zhang, Jiayuan Gu, Jilong Wang, Jingbo Wang, He Wang, Li Yi

**Published:** 2026-09-16

**Categories:** cs.RO

**Abstract:**

Humanoid robots can step over, squeeze past, and duck under obstacles, but learning to select and coordinate these behaviors from onboard perception remains challenging. Many existing approaches rely on task-specific reinforcement-learning objectives or curated motion libraries, making broad behavioral coverage costly. We present PASSAGE, a perception-conditioned planner--tracker framework for humanoid traversal. Using virtual reality and inertial motion capture, we collect 100 h of scene-aligned human motion across 1,500 cluttered scenes. A conditional flow-matching planner generates short-horizon references from motion history, a local destination, and a robot-centric multi-layer elevation map, while a perceptive whole-body tracker executes them at 50 Hz with geometric feedback. Real-time chunking promotes inter-chunk consistency, and planner-side RL post-training under the frozen tracker further improves closed-loop performance. Without skill annotations or obstacle-specific policies, one planner--tracker pair selects and composes traversal behaviors across unseen geometries. In simulation, component ablations quantify the contribution of each stage. Across three independent training seeds, scaling captured data from 6 to 100 h increases mean contact-free success from 48.1% to 68.9% on held-out scenes, while the final model with validated scene augmentation reaches 70.3%. The fully onboard system integrates egocentric 3D LiDAR perception, online occupancy mapping, 6.25 Hz planning, and 50 Hz control on a Jetson AGX Orin; tests across 50 unseen physical layouts demonstrate traversal without prebuilt maps or offboard computation.

### 论文解读
#### 摘要翻译
PASSAGE面向杂乱、未见环境中的人形机器人穿越，利用场景对齐的人体运动示范，让机器人学会跨越、下蹲和挤过障碍。系统采用感知条件规划器与跟踪器，并加入实时分块和规划器侧强化学习，在Unitree G1上实现机载感知闭环控制。

#### 方法动机分析
人形机器人虽有丰富自由度，但仅凭机载感知协调全身动作很难。任务专用强化学习或人工动作库难覆盖多种障碍。作者的关键假设是：把示范与场景几何同步记录，模型就能学习障碍形状与身体运动之间的关系；短时规划配合实时跟踪可提升泛化和执行鲁棒性。

#### 方法设计详解
输入包括Odin 3D LiDAR构建的在线占据图、运动历史和局部目标。地图编码为三层高程表示，区分障碍顶部、下方空间和支撑地面。Transformer流匹配规划器以65维运动状态为输出表示，状态含根高、重力、平面速度、偏航角速度及关节位置/速度，从噪声生成未来0.5秒的运动参考。训练同时约束流匹配、根部与关节速度、足端距离、滑移、平滑性、边界和碰撞势场。实时分块将上一段尾部与新计划渐变融合；ScaleBFM感知跟踪器以50 Hz执行并用局部几何纠偏。最后冻结跟踪器，用PPO让规划器适应执行误差。数据包含100小时、1500个程序化场景，并通过障碍尺度[0.5,1.5]、±15°旋转的变体增强。实际推理时，规划器在Jetson AGX Orin上以TensorRT FP16运行于6.25 Hz，关节目标500 Hz发布。

#### 方法对比分析
PASSAGE的区别不只是增加动作库，而是将场景几何直接作为生成运动的条件，并用RTC解决滚动生成的接缝，用感知跟踪器和规划器后训练形成闭环。它适合障碍类型多、需要连续切换穿越姿态的任务，但依赖场景对齐示范和可靠的局部几何感知。

#### 实验分析（精简版）
在150个仿真场景中，PASSAGE成功率98.67%、无接触成功率70.27%，而CAT无接触成功率仅14.00%。去掉RTC后无接触成功率降至24.5%，去掉规划器RL后降至26.4%，显示两者对安全和连续性均不可忽略。真实50次实验全部到达目标，无接触45次（90%）；透明面、细线等障碍仍可能造成接触或漏检，说明成功到达并不等于完全无碰撞。

#### 实用指南
论文提供项目页面，但本文未确认训练代码、权重和数据集是否全部公开。复现需使用MuJoCo、ROG-Map、LiDAR和G1级机器人，并保留三层地图、0.5秒窗口、TensorRT FP16下6.25 Hz规划及50 Hz跟踪设定。迁移到其他机器人必须重建状态与动力学约束，重新训练示范规划器并校准跟踪器和强化学习奖励。

#### 总结
核心思想：场景对齐示范驱动闭环穿越
1. LiDAR生成三层障碍地图。
2. 流匹配规划器生成短时全身动作。
3. RTC拼接连续计划，感知跟踪器高频执行。
4. PPO使规划器适应真实执行偏差。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.18732v1)
- [arXiv](https://arxiv.org/abs/2609.18732v1)

---

<a id='2609.19142v1'></a>
## [PointZero: 3D Point Track Completion for Learning Transferable 3D Dynamics](https://arxiv.org/abs/2609.19142v1)

**Authors:** Bardienus P. Duisterhof, Kaifeng Zhang, Adam Hung, Bowen Wen, Stan Birchfield, Yunzhu Li, Deva Ramanan, Jeffrey Ichnowski

**Published:** 2026-09-16

**Categories:** cs.CV, cs.RO

**Abstract:**

World models endow perceptual systems with the ability to predict how scenes evolve under interaction. They are most beneficial when trained on diverse volumes of data, to instill a rich prior into downstream applications. Existing methods typically require robot action labels to learn action-conditioned 3D dynamics, which excludes web video data from the training pool. We study 3D point track completion as a pre-training objective for learning transferable 3D dynamics without robot data. Given a single RGB-D observation and sparse partial 3D trajectories (tracks), we predict future 3D tracks of all observed points. We show this objective produces a rich 3D dynamics prior, without requiring robot action labels. We contribute a diverse dataset of 2.9 million synthetic frames spanning deformable, articulated, and rigid objects, and use it to train PointZero. We show that a flexible and expressive transformer, PointZero, outperforms prior methods on the same data. We demonstrate the utility of our pre-training objective by post-training PointZero for two downstream applications: (1) action-conditioned 3D dynamics prediction and (2) imitation learning. When fine-tuned to condition on end-effector pose, PointZero outperforms the baselines on the recent PGND 3D dynamics benchmark. When fine-tuned to predict robot actions and 3D tracks, PointZero outperforms or matches the baselines on 6/7 simulated and real-world robot manipulation tasks. We furthermore evaluate training PointZero from scratch to isolate the benefits of our proposed architecture from those of our proposed pre-training objective and dataset. We release the dataset, checkpoints, and full training recipe.

### 论文解读
#### 摘要翻译
PointZero提出“3D点轨迹补全”预训练任务：给定一张RGB-D图像和少量点的运动轨迹，预测场景中所有点未来的三维轨迹。它不需要机器人动作标签，可从人类演示或视频轨迹中学习可迁移的三维动力学，并进一步服务于机器人模仿学习。
#### 方法动机分析
动作条件世界模型依赖精确控制信号，难以利用互联网视频；二维预测又缺乏度量三维运动，模拟器难统一处理衣物等变形体。PointZero假设少量点的“如何动”已足以约束其余点的协同动力学，因而把动作标签依赖转化为稀疏轨迹条件。
#### 方法设计详解
RGB-D先反投影为初始点云；输入1–3个引导点在未来若干步的三维轨迹，输出全部点的密集轨迹。模型用MLP编码点坐标，把带时间和索引的稀疏轨迹编码为动作Token，并用DINOv2提取图像特征、Perceiver-IO压缩视觉Token。DiT以带噪未来轨迹和初始位置为查询，交替使用自注意力交换点间协同信息、交叉注意力读取几何、轨迹和视觉条件。作者比较回归、Flow Matching与JiT式直接预测干净坐标，后者在受物理约束的轨迹流形上表现最好。推理支持约4步Euler去噪。
#### 方法对比分析
与动作条件模型相比，它只需稀疏点轨迹；与二维视频模型相比，它直接预测度量三维坐标；与图网络粒子模拟相比，它结合视觉上下文和扩散生成，能覆盖刚体、关节体及非刚体运动。适合短时、局部观测的交互预测，长程滚动和严重遮挡仍需谨慎。
#### 实验分析（精简版）
合成集约含290万帧，覆盖毛巾、衣物、门、抽屉及折叠、推拉等交互。变形体上，PointZero-JiT的平均距离误差为2.85 cm，PGND为9.01 cm。真实评估含14种物体、124次交互，模型在12项指标中11项排名第一；用于7个机器人任务时，预训练使平均成功率由80.5%升至88.2%。这些结果支持零样本迁移和下游表示价值，但尚不足以证明长时闭环稳定性。
#### 实用指南
论文提供代码、数据集和预训练权重，训练报告为8张H100约2天。复现时要严格保持RGB-D尺度、点轨迹索引和时间编码，并评估MDE、MSE、CD、EMD，同时检查坐标单位与遮挡处理。迁移到新机器人需把引导轨迹换成末端或抓取点轨迹，并训练动作头；新相机、物体或更长时域则需重新验证和微调。
#### 总结
核心思想：稀疏轨迹补全三维动力学
速记：
1. RGB-D生成带尺度的初始点云，并选取少量引导点。
2. 编码视觉、几何与带时间索引的稀疏轨迹条件。
3. DiT交替融合点间关系和条件，扩散去噪补全所有点的未来运动。
4. 将预测轨迹接入模仿学习，输出机器人控制策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.19142v1)
- [arXiv](https://arxiv.org/abs/2609.19142v1)

---

<a id='2609.18514v1'></a>
## [ActiveScale: Scaling Active Perception for Robots across Model, Data, and Hardware](https://arxiv.org/abs/2609.18514v1)

**Authors:** Shuai Zhou, Kaisheng Pang, Wenxuan Song, Wenjie Zhang, Xinhu Zheng, Haoang Li

**Published:** 2026-09-16

**Categories:** cs.RO, cs.LG

**Abstract:**

Active perception is essential for robotic manipulation when fixed viewpoints leave task-relevant information occluded or unobserved. However, enabling vision-language-action (VLA) models to reason across changing viewpoints and actively acquire informative observations remains challenging. We present ActiveScale, a framework that advances active perception through coordinated model, data, and hardware designs. Our model augments a VLA with historical video observations and explicit camera-pose supervision, using per-frame pose tokens and a lightweight prediction head to associate observations across viewpoints and support a coherent understanding of the scene. To learn from the camera motion naturally present in human activity, we introduce a scalable human--robot mid-training recipe using 1000 hours of egocentric and robotic data, adapting the model to temporal inputs and pose supervision. We further introduce Active-perception Mobile-manipulation Platform (AMP), a robotic platform that supports active perception and mobile manipulation through single-operator teleoperation, enabling scalable collection of demonstrations that coordinate viewpoint changes and manipulation. Experiments demonstrate improved success rates on active-perception tasks, while ablation studies validate the contributions of camera-pose-aware modeling and egocentric mid-training. Together, these components provide an integrated foundation for studying and developing active perception in robotic manipulation.

### 论文解读
#### 摘要翻译
ActiveScale 研究如何从模型、数据和硬件三方面扩展机器人的主动感知。作者构建 AMP 主动感知移动操作平台，利用人类第一人称与机器人数据训练基于 π0.5 的视觉语言动作模型，使机器人能主动改变相机视角来处理遮挡和搜索任务。

#### 方法动机分析
固定视角 VLA 看不到包、抽屉、桌下或高低不同货架中的目标；简单加入历史画面也难以知道画面变化对应怎样的相机运动。论文的关键假设是，人类第一人称数据天然记录了“为获得信息而移动视点”的搜索先验，显式学习相机位姿可以把跨视角观测联系起来。

#### 方法设计详解
模型输入当前画面、间隔 16 帧采样的 3 帧历史画面、机器人状态和语言指令。每帧视觉表示后追加可学习 camera token，并用 block-causal attention 让动作生成模块读取完整视角历史。camera token 经过预测头回归 9 维相机状态：三维位置、四元数姿态和水平/垂直视场角；相机损失由平移、旋转及 FOV 项组成，FOV 权重为 0.5。策略同时输出左右臂末端位姿、夹具和主动相机动作，训练结合 FAST 离散动作损失与 flow matching。AMP 在移动底盘上配置左右 6-DoF Piper 操作臂，并以中央 6-DoF Piper 机械臂携带 Orbbec DaBai DC1 深度相机；Quest 2 头显控制相机，双手柄控制双臂和底盘。先用约 1000 小时人类—机器人数据中期训练，再以每任务族约 150 条演示微调。

#### 方法对比分析
相较 π0.5 等从当前视角直接操作的方法，ActiveScale 的创新一是把相机纳入动作空间，二是用位姿监督解释不同画面的几何关系，三是引入人类第一人称搜索轨迹进行共同训练。相较只拼接历史图像，camera token 能表达“从哪里看”；相较纯软件插件，AMP 提供可执行的独立视角自由度。它尤其适合遮挡、视场外目标和主动定位，但依赖可控相机机构、轨迹数据与精确标定。

#### 实验分析（精简版）
在 Bag、Drawer、Table、Pot、Box 五个真实任务上，平均成功率由 π0.5 的 30.0% 提升至 70.0%，任务进度由 41.6% 提升至 78.4%。Table 任务中目标初始完全不可见，成功率从 10% 提升到 80%。去除中期共同训练后，成功率降至 62.0%、任务进度降至 67.7%，说明人类搜索数据重要。单 RTX 4090 上动作生成达 264.5 Hz，满足 30 Hz 控制；但长程导航和极端光照下的泛化仍未充分验证。

#### 实用指南
项目网页为 active-scale.github.io，论文说明将发布推理代码、预训练权重和 AMP 设计文件，使用前需核验最新状态。复现要准备带相机轨迹的人类第一人称数据、机器人数据、共享坐标系标定和 4 帧历史输入。迁移到其他机器人时，需要重做相机标定、动作映射，并用目标平台演示适配相机与动作头。

#### 总结
核心思想：主动换视角获取关键信息
1. 独立相机臂执行视角动作。
2. 位姿 token 对齐历史视觉。
3. 人类搜索先验与机器人数据共同训练。
4. 平台演示微调完整移动操作策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.18514v1)
- [arXiv](https://arxiv.org/abs/2609.18514v1)

---

<a id='2609.18359v1'></a>
## [RecMorph: Topology-Guided Spatial Recurrence for Generalized Morphology Control](https://arxiv.org/abs/2609.18359v1)

**Authors:** Quanrui Rao, Yong Liu, Xueming Xiao, Yingbo Luo, Kun Wu, Zhenyu Xu, Meibao Yao

**Published:** 2026-09-16

**Categories:** cs.RO, cs.LG

**Abstract:**

Generalized morphology control requires a single policy to transform information across limbs with different physical roles, coordinate whole-body motion, and remain efficient as body size grows. Existing communication mechanisms address these requirements only partially. We introduce RecMorph, a topology-guided spatial recurrent architecture that uses recurrent sequence computation to jointly perform cross-limb communication and representation transformation. A depth-first traversal converts the kinematic tree into a morphology-derived sequence, along which shared bidirectional transitions progressively transform limb information before action decoding. Residual preservation, RMS normalization, and input-dependent channel modulation stabilize this repeated spatial transformation, yielding linear token complexity at fixed model width and depth. Across five UNIMAL tasks, RecMorph achieves the strongest mean final training performance among the evaluated generalized morphology controllers and the highest measured inference throughput on FT, while generalizing to unseen variations and bodies with up to 30 limbs. We further migrate representative generalized controllers from UNIMAL benchmarks to a four-platform quadruped setting. RecMorph achieves the best macro-averaged performance under nominal and high friction, reduces nominal velocity RMSE by 43.5% relative to specialist MLPs, and one shared policy completes 40 physical Go1/Go2 trials without falls. These results show that topology-guided recurrent transformation provides an effective and efficient communication mechanism for Generalized Morphology Control and remains effective when transferred from procedural bodies to physical robot platforms. Code and experimental resources are publicly available at https://github.com/quanruirao/RecMorph.

### 论文解读
#### 摘要翻译
RecMorph 面向泛化形态控制：同一策略要在不同物理结构的肢体间转换信息、协调全身，并随肢体数增加保持效率。它把运动学树按深度优先遍历（DFS）变成序列，以共享双向空间递归逐步转换肢体表示，再解码动作。残差、RMS 归一化和输入依赖的通道调制稳定递归计算，使令牌复杂度线性增长。方法可泛化至未见形态和最多 30 个肢体，并迁移到四种四足机器人。

#### 方法动机分析
不同质量、齿轮比和拓扑会改变同一局部动作对全身运动的含义。图消息传递需要多轮扩散，可能稀释局部信息；全自注意力虽能直接通信，却有 O(N²) 成本，且偏重聚合，缺少显式的目标肢体语境转换。论文假设 DFS 能保留运动学子树结构，双向扫描即可让每个肢体获得全身上下文。

#### 方法设计详解
输入是各肢体的本体感受观测与形态属性，经共享编码器变成令牌；运动学树通过 DFS 前序遍历序列化。4 个双向循环块沿序列从两端扫描：正向和反向隐状态分别汇总两侧信息，再合并为全局肢体上下文。每层使用 RMSNorm 稳定数值，用残差保留局部表示，并由当前肢体生成通道门控，对传来的上下文逐通道筛选，避免无关信息覆盖本地特征。地形任务还融合高度场，最后由共享解码器为激活执行器输出动作。策略训练采用 PPO，UNIMAL 训练约 1 亿次环境交互、四足实验使用 10,000 次迭代；论文配置嵌入宽度 128、循环隐层 256、深度 4。该扫描的令牌成本随肢体数线性增加。

#### 方法对比分析
相较 MetaMorph，RecMorph 用拓扑序列上的双向递归取代全注意力，降低复杂度并显式完成形态语境变换；相较 NerveNet，它用一次空间扫描覆盖全身，而非依赖多轮邻域消息传递。论文的创新贡献是把 DFS 拓扑引导、共享双向转换和通道门控结合为稳定的空间递归机制，而不只是替换一个聚合模块。它适合形态和肢体规模变化大、又要求低延迟的控制器，但需要正确的运动学拓扑与属性编码。

#### 实验分析（精简版）
在包含平地、斜坡、探索、复杂地形和障碍物的五个 UNIMAL 任务中，RecMorph 获得最高平均最终训练性能；斜坡任务比次优方法高 48.1%，FT 推理吞吐量约 1800 FPS。扩展到 30 个肢体时仍优于 Transformer 类方法。迁移到 Go1、Go2、ANYmal-B、ANYmal-C 后，名义速度 RMSE 相对专家级 MLP 降低 43.5%，40 次真实 Go1/Go2 试验均无跌倒。消融中去除残差、RMSNorm 和通道调制使性能由 4367 降至 2188，说明稳定化设计不可忽略；低摩擦和更长链条形态仍是主要局限。

#### 实用指南
论文提供的代码与资源链接为 https://github.com/quanruirao/RecMorph。复现需按运动学树生成 DFS 顺序，准确实现本体感受和形态属性输入，并使用 MuJoCo 的 UNIMAL 环境；PPO 超参数应参考论文附录。迁移到其他机器人要重做拓扑、属性、动作映射和动力学/摩擦配置，通常还需平台数据适配。论文报告 UNIMAL 约 100 个形态、约 1 亿次环境交互；其余工程依赖和完整部署脚本需以仓库为准。

#### 总结
核心思想：拓扑引导的双向空间递归
1. DFS 序列化运动学树。
2. 双向递归传播全身上下文。
3. 门控、归一化和残差稳定形态变换。
4. 共享策略输出动作并迁移真实四足。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.18359v1)
- [arXiv](https://arxiv.org/abs/2609.18359v1)

---

<a id='2609.18542v1'></a>
## [Accuracy- and Real-Time-Aware 4D Radar Preprocessing for Autonomous Driving Perception Systems](https://arxiv.org/abs/2609.18542v1)

**Authors:** Woo-Jin Jung, Dong-Hee Paek, Jeong-Su Park, Seung-Hyun Kong

**Published:** 2026-09-16

**Categories:** cs.CV

**Abstract:**

4D radar has emerged as a promising next-generation sensor for improving the robustness of autonomous driving perception systems because of its stable sensing capability under adverse weather conditions. However, deploying 4D radar in embedded environments with limited hardware resources requires radar-representation preprocessing that jointly considers perception accuracy, real-time performance, and computational complexity. This paper proposes a preprocessing framework for 4D-radar-based 3D object detection. First, Percentile-based 3D Shape Preservation (P3DP) extracts point clouds from radar tensors while preserving object-shape information and suppressing noise and false alarms. Second, Multi-frame-based Noise Point Discrimination using Kernel Density Estimation (MF-KDE) improves the density and reliability of sparse radar point clouds. Finally, Embedded \& NetScore (ENS) evaluates suitability for embedded deployment by jointly considering accuracy, real-time performance, adverse-weather robustness, and model complexity.

### 论文解读

#### 摘要翻译
4D 雷达耐受雨雾等恶劣天气，但高维张量计算昂贵；普通点云又太稀疏。论文提出 P3DP 保留张量中的目标形状，MF-KDE 用多帧核密度增强点云，并以 ENS 联合衡量精度、实时性和模型复杂度。

#### 方法动机分析
关键问题是如何在车载算力有限时保留真正有用的几何证据。高功率单元可能来自旁瓣和干扰，简单取峰值会把噪声送进检测器；单帧雷达点则缺乏密度。作者假设目标边界具有明显功率变化，且真实目标会在连续帧中重复出现，因此局部密度可用于区分目标与孤立噪声。

#### 方法设计详解
P3DP 先把张量裁剪到纵向 0–73 m、横向 ±16 m、垂向 −2–6 m，再按距离平方补偿功率：\(P_{norm}=r^2P\)。从归一化结果中取功率前 10% 的候选单元，用 Difference of Gaussians 估计局部功率密度，并保留密度较低的底部 50%，以突出目标表面和边界，最后交给 RTNH、RadarPillar-Net 等检测器。MF-KDE 将此前 5 帧对齐到当前坐标系，对点计算 KDE 密度 \(\rho\)，把输入特征扩为 \([x,y,z,v,r,\rho]\)，让网络学习稳定目标点与孤立噪声的差异。ENS 将正常场景 AP、恶劣天气 AP、nuScenes Detection Score 放在分子，将参数量和 MAC 放在分母，权重为 \(\alpha=2,\beta=\gamma=0.5\)。因此，P3DP 通过减少无效体素降低稀疏卷积负担，MF-KDE 则以时间一致性弥补单帧稀疏；两者分别作用于张量和点云输入，可组合使用的范围取决于传感器输出格式。

#### 方法对比分析
相较 CA-CFAR 和固定百分位筛选，P3DP 使用功率变化而非幅值本身来保留结构，并针对旁瓣、均匀干扰和目标边界作区分；相较单帧点云或 DoppDrive，MF-KDE 将多帧空间一致性变成显式密度特征，帮助网络处理切向运动造成的错配。本文的创新在于把形状保持、密度增强和嵌入式评价统一到雷达预处理链路中。ENS 也不只看 AP，而是把天气鲁棒性、3D 质量和嵌入式成本纳入评价，适合实时 3D 雷达检测；但静态阈值和多帧对齐会依赖传感器标定与运动补偿。

#### 实验分析（精简版）
K-Radar 用于张量实验，Dual Radar 用于点云实验，测试在 RTX 3090 上进行，并比较 RTNH、RadarPillar-Net、RPFA-Net 和 MF-Net。RadarPillar-Net+P3DP 获得 65.86% BEV AP 和 64.43% ENS；RTNH 的 FPS 从 22.65 提升到 25.60，3D AP 相对标准百分位方法约提升 4%。MF-Net+MF-KDE 达到 39.21% 3D AP、54.17% ENS。优势是质量与效率兼顾；局限是 KDE 增加 \(O(M\log M+MK)\) 计算，GPU 结果不等于目标车载 SoC 的端到端延迟，且文中未给出完整端到端车载功耗证据。

#### 实用指南
论文使用公开 K-Radar 和 Dual Radar，但未在文中明确给出代码或预训练模型链接。复现需保持 ROI、10% 候选、底部 50% 筛选、5 帧对齐和 KDE 特征，并按论文权重计算 ENS。迁移到其他雷达时应重新标定功率、调整坐标范围和 KDE 带宽，重训检测器，并在实际 SoC 上测量预处理与稀疏卷积延迟；必要时可把 KDE 并行化到 FPGA/DSP。

#### 总结
结构化密度让雷达兼顾精度与实时
1. 距离补偿并筛出高功率候选。
2. 用 DoG 保留有结构的边界点。
3. 对齐 5 帧并加入 KDE 密度特征。
4. 用 ENS 评估精度、鲁棒性和部署成本。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.18542v1)
- [arXiv](https://arxiv.org/abs/2609.18542v1)

---

<a id='2609.18326v1'></a>
## [UAVs Meet Embodied Intelligence: Bridging Human Intents and Flying Dynamics Via Harnessing Physical-Digital AI Agents](https://arxiv.org/abs/2609.18326v1)

**Authors:** Yonglin Tian, Weiyi Wang, Houhua Lu, Xinyi Li, Yihao Wu, Jingyang Chen, Jianli Sun, Chengxiang Li, Yinuo Chen, Fei Lin, Tengchao Zhang, Jing Yang, Deyi Ji, Jian Di, Naiqi Wu, Yisheng Lv

**Published:** 2026-09-16

**Categories:** cs.RO

**Abstract:**

Unmanned aerial vehicles (UAVs) extend embodied intelligence into continuous three-dimensional space, where perception, reasoning, physical embodiment, and action are tightly coupled through flight and environmental interaction. Recent advances in foundation models, world models, and AI agents are shifting UAV autonomy from task-specific perception and control toward systems that can interpret human intent, understand open environments, reason about physical consequences, and organize complex behaviors under embodiment and flight-dynamic constraints. We characterize this emerging paradigm as UAV embodied intelligence (UAV EI) and distinguish it from its system realization, the embodied-intelligent UAV (EI UAV). To provide a unified view of the field, we introduce a 5+5 framework that describes UAV EI through five capability dimensions and EI UAVs through five architectural layers spanning physical embodiment, general cognition, embodied skills, external interaction, and system harnessing. Based on this framework, we systematically review recent progress in embodied morphology, embodied perception, world models, embodied planning, vision-language navigation, embodied manipulation, and embodied collaboration. We further identify long-horizon autonomy, predictive physical reasoning, test-time skill acquisition, and autonomous capability evolution as key challenges toward more general aerial embodied intelligence. Finally, we argue that harnessing physical-digital AI agents, through persistent coupling of digital intelligence with physical sensing, dynamics, action, and feedback, provides a system-level pathway toward adaptive and continuously evolving UAV autonomy. Project resources are available at our project website and GitHub repository.

### 论文解读

#### 摘要翻译
本文讨论无人机具身智能（UAV EI）：无人机应能理解开放式人类意图，在三维环境中推理，并将决策落实到受飞行动力学约束的行动。作者提出“5+5”框架，梳理能力维度与系统架构，并以物理—数字AI智能体连接通用认知和飞行身体。

#### 方法动机分析
传统无人机多针对固定任务分别设计感知、规划和控制模块，面对自然语言指令、未见环境、长时任务及天气和电量变化时泛化不足。论文的核心假设是，大模型可以承担语义理解与推理，但不能脱离机体状态和动力学直接控制；因此需要世界模型、具身技能、闭环反馈和安全运行时共同完成落地。

#### 方法设计详解
能力侧的五项是人类意图理解、具身自我意识、通用环境理解、复杂任务规划和闭环执行；实现侧的五层是具身层、大脑层、小脑层、交互层和 Harness 层。任务输入包括语言、视觉/深度观测、状态估计及机体约束。大脑层把意图转成任务表示，结合时空环境理解和世界模型预测动作后果；具身感知还会主动调整视点、距离和姿态以获取信息。规划可采用几何/动力学优化、观测到动作的策略，或带任务分解、记忆和反思的 Agentic planning。Digital UAV Agent 负责推理、记忆和规划，Physical UAV Agent 负责感知、控制和交互；Harness 管理任务状态与执行流程，检查安全约束并把高层计划交给导航、操作等小脑技能，再将飞行反馈回传。

#### 方法对比分析
论文将世界模型中的 MAD、WorldVLN、AirDreamer 等九类系统按任务、连续控制或六自由度航点动作、真实零样本或仿真部署进行比较；又归纳十五种规划方法和二十五种协作方法，区分状态/RGB输入、控制/航点抽象，以及 HOCBF、奖励塑形等可行性处理。本文的区别在于提供跨模块的系统框架，而非提出一个替代所有基线的控制器，适用于开放任务、主动感知和异构协作的架构设计。

#### 实验分析（精简版）
本文主要证据来自文献归纳和结构化对比，不是统一数据集上的新模型实验；因此没有可据此宣称的统一准确率、成功率或延迟提升。比较范围明确包含世界模型9类、规划15种和协作25种方法，但这些数量是综述覆盖范围，不是性能分数。主要优势是建立能力—架构映射，局限是缺乏统一端到端基准，且长时程、物理推理和高保真空中操作数据仍不足。

#### 实用指南
复现应先依据被引用工作的原始论文核对数据、依赖和训练设置；本文未统一给出 batch size、学习率、epoch、GPU 或实时延迟。实现时可让大脑层生成结构化计划，由小脑层执行动力学可行技能，并让 Harness 负责状态同步、失败重规划和危险动作拦截。换机型需重做机体参数、控制接口与技能库，换环境需验证感知和世界模型分布。论文提及项目网站与 GitHub 资源库，但未在本文中完整提供一个端到端可运行实现。

#### 总结
核心思想：让无人机理解意图并受物理约束行动。
1. 语言与观测被转成结构化任务；
2. 主动感知和世界模型预测环境及飞行后果；
3. 数字智能体分解任务并调用具身技能；
4. Harness 检查安全约束，物理智能体闭环执行并反馈。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.18326v1)
- [arXiv](https://arxiv.org/abs/2609.18326v1)

---

<a id='2609.18504v1'></a>
## [InterMASH: A Unified Geometric Representation for Grasp Synthesis](https://arxiv.org/abs/2609.18504v1)

**Authors:** Xuanze Yang, Yumeng Liu, Haiyang Xin, Changhao Li, Haowei Shen, Kai Xu, Ligang Liu, Ruizhen Hu

**Published:** 2026-09-16

**Categories:** cs.RO, cs.GR

**Abstract:**

Grasp synthesis aims to generate stable and physically plausible hand--object interactions, and has become a fundamental problem in both human hand modeling and robotic manipulation. However, a unified representation across human and robotic hands is still lacking, mainly due to differences in hand morphology and surface modeling. Prior methods typically rely on either contact maps or dense implicit descriptors to represent interaction, but these representations are often incomplete or computationally expensive and redundant. We propose InterMASH, a unified geometric representation that establishes cross-embodiment correspondence using sphere-fixed anchors. At each anchor, low-degree spherical harmonics compactly encode local hand geometry, object geometry, and contact, forming an explicit and interpretable token sequence. Building on this natively tokenized structure, we introduce a conditional Diffusion Transformer that operates directly in the proposed InterMASH representation space and jointly generates hand geometry and contact, improving consistency and physical plausibility. Our method achieves competitive performance with state-of-the-art methods on key physical feasibility metrics in a large-scale ShadowHand benchmark, supports joint training across multiple hands, and shows that cross-embodiment fine-tuning with human grasp data can improve robotic grasp success and diversity. Project page is available at https://inter-mash.github.io/.

### 论文解读
#### 摘要翻译
InterMASH 面向人手与机器人手交互表示不统一的问题：它在物体周围的球面上布置固定锚点，以低阶球谐编码手、物体和接触的局部几何，再用条件扩散 Transformer 联合生成手部形状与接触。人类和机器人数据的联合训练可提升机器人抓取的成功率与多样性。

#### 方法动机分析
不同手型的网格拓扑、自由度和数据分布差异很大，单一手型生成器难以迁移；全局点集又不易保留局部接触结构。论文的关键假设是，物体中心球面提供稳定的空间索引，而锚点局部 patch 能在不同具身间共享几何语义。该表示主要服务于三维抓取合成，仍依赖手模板和跨手语义对齐。

#### 方法设计详解
给定物体和模板手，方法以物体为中心、半径 0.2 的球面用 Fibonacci 采样放置锚点。每个锚点 token 同时记录位置、局部方向、手/物体的 SH 系数与视觉掩码参数，并附带接触描述。局部表面由 SH 基函数加权重建，实验通常使用 128 个锚点、最高二阶 SH。ShadowHand、MANO、Barrett、Allegro 等模板先在关键点距离特征上通过线性指派问题对齐。随后条件扩散 Transformer 以物体和模板手为条件生成手部局部几何及接触；近邻增强注意力把 k 近邻图形成的偏置加入注意力，使相邻 patch 保持几何联系。反向扩散还利用穿透、自碰撞和接触稳定性惩罚的梯度进行物理引导，最终通过保持锚点语义的分块逆运动学拟合输出关节姿态。训练使用 PyTorch、Muon 优化器，并将 SH 阶数从 0 阶逐步提升到 2 阶。

#### 方法对比分析
InterMASH 将 BPS 的固定空间索引与 MASH 的局部几何描述结合，不只是对点云进行统一采样；它还显式建模接触，并通过模板关键点对齐跨手语义。相比普通 Transformer，邻域偏置更关注局部结构；相比无约束生成，物理梯度直接参与采样。因而它适合多手型抓取、人类先验迁移和跨具身训练，但通用表示对高自由度手的专门归纳偏置较弱，小数据下可能不够高效，左右手混训也有 handedness 歧义。

#### 实验分析（精简版）
在 DexGraspNet 的 ShadowHand 评测中，InterMASH 的 Suc.1 为 91.9，高于 DGA 的 90.4；穿透指标为 16.2，低于 DGA 的 21.5。CMapDataset 上，Shadow+Barrett 联合训练使 Barrett 成功率达到 90.30；加入邻域增强注意力后，该指标由 85.20 提升到 90.30。去掉物理引导，Shadow 成功率由 64.15 降至 51.06，说明物理约束确实重要。人类先验微调使 Suc.6 从 25.8 提升至 29.0，但穿透由 14.9 增至 18.6，效果存在权衡。

#### 实用指南
复现需实现球面采样、SH patch 编码、跨手关键点指派、近邻注意力、物理惩罚和 patch-wise IK，并遵循 DexGraspNet、CMapDataset、DexGRAB/GRAB 的数据协议。论文给出项目页 https://inter-mash.github.io/；代码是否已公开、依赖版本和完整超参数未说明。迁移到新手型要替换模板、重新对齐并训练或微调；迁移到新物体域则需保持锚点编码并重新校准物理损失。

#### 总结
核心思想：锚点球谐统一跨手抓取

速记 pipeline：
1. 球面锚点固定物体周围空间索引。
2. SH token 联合表示局部手、物体与接触。
3. 线性指派对齐不同手型语义。
4. 扩散 Transformer 加邻域注意力生成抓取。
5. 物理引导采样并用 IK 输出姿态。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.18504v1)
- [arXiv](https://arxiv.org/abs/2609.18504v1)

---

<a id='2609.18930v1'></a>
## [Learning Holistic Whole-Body Loco-Manipulation with a Bipedal Mobile Manipulator](https://arxiv.org/abs/2609.18930v1)

**Authors:** Zhongyu Chen, Yuxuan Nai, Qian Chen, Yidong Zhu, Chen Jing, Qihan Wang, Xudong Li, Zhizhan Li, Leixin Chang, Liangjing Yang, Hua Chen

**Published:** 2026-09-16

**Categories:** cs.RO

**Abstract:**

Bipedal loco-manipulation enables robots to interact with objects beyond the nominal workspace of their arms by coordinating locomotion and manipulation. Realizing this capability requires a low-level whole-body controller that translates task-level manipulation goals into coordinated arm and leg motions while maintaining balance. We present a unified whole-body controller trained with reinforcement learning that directly maps 6-DoF end-effector targets to coordinated actions for the bipedal base and robotic arm. Given only an end-effector target, the learned controller autonomously coordinates reaching, postural adaptation, and stepping without explicit base-velocity or footstep commands. A reward-gating strategy regulates the trade-offs among end-effector tracking, locomotion, and balance during training, while a temporal context estimator combines windowed Transformer encoding, recurrent GRU memory, and auxiliary dynamics prediction to extract dynamics-relevant information from observation history. Real-robot experiments demonstrate that the same controller supports reaching, postural adaptation, and stepping under commands from VR teleoperation, a learned diffusion policy, and scripted trajectories, providing a common end-effector interface for diverse manipulation tasks.

### 论文解读
#### 摘要翻译
论文提出一个面向双足移动操作机器人的统一全身控制器：上层只需提供六自由度末端执行器目标，系统便能自主协调机械臂、腿部姿态和迈步，不要求额外指定基座速度或足迹。方法在 Isaac Lab 中用强化学习训练，并部署到 LimX TRON 1 双足机器人与 ARX L5 六自由度机械臂，支持 VR 遥操作和扩散策略。

#### 方法动机分析
驱动力是扩大固定基座机械臂有限的作业空间；现有方法的痛点是将行走、平衡和末端跟踪分开，并要求高层显式规划底盘运动。双足平台尤其容易出现手臂动作破坏平衡、行走降低操作精度的问题。核心假设是：整体策略可根据末端目标距离自主决定伸展、下蹲或迈步，并用历史观测补足单帧状态无法直接观测的速度和动力学信息。

#### 方法设计详解
策略输入本体观测、末端目标和时间上下文，输出 14 个腿臂关节的位置增量，再叠加默认姿态形成期望关节位置。时间上下文估计器读取过去 10 帧，使用 Transformer 捕捉短期变化、GRU 保留长期信息，并辅助预测局部基座线速度和动力学潜变量。奖励端引入双足感知门控：远离目标时偏重稳健移动，接近目标时偏重精确跟踪；best-so-far 进度奖励只奖励刷新历史最小误差，减少振荡。训练设定采用 PPO、8192 个并行环境；推理设定为策略 50 Hz、底层 PD 500 Hz。

#### 方法对比分析
本方法的本质区别与创新是：相较 floating-base+IK，它直接学习末端目标到腿臂动作的闭环映射，不需要上层给出脚步序列；相较无历史潜变量的策略，Transformer-GRU 显式利用时间信息；相较固定权重奖励，距离相位门控能在移动和操作之间动态切换。其价值是统一接口和全身协调，适合需要扩大作业空间、又希望接入 VR 或学习型高层策略的双足平台。

#### 实验分析（精简版）
奖励门控将成功率从 82.73% 提升至 88.30%，平均位置误差从 3.23 cm 降至 2.85 cm；无潜变量配置成功率为 69.87%，Transformer-GRU 配置的最低动作变率为 0.165。真机垂直可达范围达到 3–191 cm，而 floating-base+IK 基准为 38–163 cm。系统完成了拾取玩具、擦黑板等 VR 任务及扩散策略控制，但动态跟踪精度和极端动态操控仍有限。

#### 实用指南
复现时需实现论文给出的 58 维观测、奖励项和双足动力学，并使用 Isaac Lab/PPO 训练。迁移到其他机器人要重新配置关节映射、默认姿态、动作尺度、碰撞参数和安全奖励，通常还需重新训练上下文估计器。论文介绍了硬件和训练设置，但未明确确认完整代码或权重公开，不能直接假定已有开源实现。

#### 总结
核心思想：末端目标驱动双足全身协调

速记：
1. 观测与目标输入。
2. Transformer-GRU 推断动态上下文。
3. 距离门控移动/操作奖励。
4. PPO 输出腿臂动作并由高速 PD 执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.18930v1)
- [arXiv](https://arxiv.org/abs/2609.18930v1)

---

<a id='2609.19138v1'></a>
## [In-Context Robot Learning with VLM Agents](https://arxiv.org/abs/2609.19138v1)

**Authors:** Dongzhou Cheng, Taoran Yi, Ye Fang, Xingwu Zhang, Fan Feng, Yixuan Li, Gengxiong Zhuang, Rongze Wang, Shuai Yang, Wei Song, Weizhi Xue, Minyan Wu, Jie Gui, Jiaqi Wang, Tong Wu

**Published:** 2026-09-16

**Categories:** cs.CV, cs.RO

**Abstract:**

Enabling robots to adapt to unfamiliar environments as readily as humans remains a moonshot goal of embodied AI. No finite collection of demonstrations can cover every task and situation a robot will encounter, making the ability to learn from context at deployment essential for generalization. Such in-context learning (ICL), however, remains largely beyond the reach of existing robotic policies. The broad agentic capabilities of commercial vision-language models (VLMs), such as GPT-6 Astra, raise a compelling question: can these models learn from demonstrations, examples, and interaction feedback, then translate that information into executable and verifiable robot behavior from a new initial state without gradient updates or persistent changes to task-specific parameters? We introduce GPT-Policy, a general-agent framework for in-context robot learning. GPT-Policy integrates a context compiler that preserves task-relevant visual transitions, a VLM that proposes robot-tool actions, and a constrained controller that verifies and executes each action and reports its outcome. We evaluate its reliability and limitations through task success and efficiency metrics, matched comparisons across models, and controlled context ablations. In real-robot trials, human video demonstrations improve task completion even without robot action labels, while aligned action references yield further gains on contact-sensitive tasks. These findings position GPT-Policy as a step toward robot adaptation through in-context learning, providing an empirical foundation for translating the general-purpose capabilities of VLMs into physical behavior and clarifying the challenges that must be overcome for reliable deployment.

### 论文解读
#### 摘要翻译
机器人面对新任务、陌生摆放和意外交互时，有限演示无法覆盖所有情况。论文提出 GPT-Policy，研究固定参数的通用视觉语言模型（VLM）能否在不梯度更新的情况下，从演示、目标图像和交互历史学习并执行新任务。系统将上下文编译为模型输入，由 VLM 提出机器人工具动作，再由约束控制器验证、执行并反馈结果。

#### 方法动机分析
传统机器人策略依赖任务专用数据和训练，部署时泛化受限。作者希望把 VLM 的上下文学习能力引入物理世界：目标图像说明“做到什么”，视频说明“如何做”，动作记录补充运动细节，历史和人类反馈则支持探索与纠错。关键假设是，清晰保留视觉变化和时序关系的上下文，足以帮助固定 VLM 在新状态中选择合适动作；但高层理解并不自动保证安全接触和准确完成。

#### 方法设计详解
输入包括任务指令、带视角标签的当前图像与机器人状态，以及目标图像、抽取关键帧的人类/机器人视频、可选的时间对齐动作和在线历史。上下文编译器交错组织图像与文字，并提供本体、坐标系和工具 schema。固定 VLM 输出 move_to 或连续 move_eef_chunk 等工具请求，而不是直接输出关节轨迹。笛卡尔适配器对位置线性插值、姿态作最短弧 SLERP；每个采样位姿以此前 IK 解为种子求逆运动学并检查位置、姿态残差，之后用 Ruckig 加入速度、加速度和 jerk 约束。执行层返回测量状态、端点误差及拒绝/完成反馈，驱动下一轮重规划。双臂请求可用 null 保持某臂姿态，夹爪由独立工具控制。

#### 方法对比分析
不同于需要机器人专门训练或测试时更新参数的策略，GPT-Policy 的核心创新是用统一 context-to-action 接口考察通用 VLM 的即时适应。人类视频可跨形态传递操作策略；机器人视频加动作则减少关键帧之间的轨迹歧义；目标图像直接表达空间布局。插值、IK 和轨迹定时是执行适配，而上下文编译、工具反馈闭环是方法贡献重点。相比只接受文本或离散技能标签的基线，这种组合同时保留视觉时序和连续位姿信息；它适合需要推理、重规划和多模态参考的任务，不适合让高延迟 VLM 独自承担快速力控。

#### 实验分析（精简版）
真实机器人每个条件重复三次，并按最终几何/语义状态判定成功。人类视频使“捡红毛巾”成功率从 0/3 提升至 2/3，平均决策从 96.3 降到 76.7，时间从 24.6 降到 18.9 分钟；机器人视频加动作使“拧瓶盖”达到 3/3，而无上下文为 0/3。目标图像、自历史和在线人机交互任务均报告 3/3。局限是样本小、条件有限，且出现双臂碰撞、接触失败、结果验证不足和较高决策延迟；这些结果证明上下文有帮助，但不等于已实现可靠通用技能。

#### 实用指南
论文提供代码仓库和项目网站。推理时 VLM 参数保持固定，不进行梯度更新；每次工具执行后把观察和反馈送回下一次决策。复现需保持场景重置、三次试验、成功判据、视频关键帧与动作的时间对齐，以及机器人基座坐标和 xyzw 四元数约定。迁移到新机械臂要重做工具适配、标定、IK、轨迹约束和安全阈值；迁移任务则要重新准备目标/演示上下文并定义终态。部署建议增加双臂联合碰撞检查、力或滑移感知及快速低层控制器。

#### 总结
核心思想：**用上下文驱动机器人闭环适应**
1. 编译演示、目标和历史。
2. VLM 生成工具动作。
3. 约束控制器检查并执行。
4. 根据反馈重规划并验证终态。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.19138v1)
- [arXiv](https://arxiv.org/abs/2609.19138v1)

---

<a id='2609.18910v1'></a>
## [CaSCo: Cascade-Aware Soft-Collision Motion Planning](https://arxiv.org/abs/2609.18910v1)

**Authors:** Shivaram Kumar, Gaoyuan Liu, Yoonchang Sung

**Published:** 2026-09-16

**Categories:** cs.RO

**Abstract:**

Conventional motion planning treats collision as a binary constraint, although contact with different objects can have drastically different consequences. A robot may safely brush against a cardboard box while even minor contact with a glass, laptop, or unstable object may be undesirable. Moreover, a direct robot--object collision can move the contacted object and trigger secondary object--object collisions, making the risk of a motion depend on the physical evolution of the scene rather than only on the robot's geometric path. We present CaSCo, a cascade-aware soft-collision motion planning framework in which a vision-language or language model assigns semantic risk to objects and a physics simulator predicts the consequences of candidate robot motions. CaSCo searches for a path that minimizes the total semantic risk of the unique objects displaced either directly by the robot or indirectly through cascaded collisions. Because collisions change the environment, we augment roadmap states with the predicted object arrangement and the set of objects whose risk has already been incurred. We develop an optimal graph-search algorithm with an admissible and consistent cascade-relaxed heuristic and caching and pruning mechanisms for efficient search. Experiments in cluttered manipulation environments evaluate semantic risk, cascade reasoning, planning efficiency, and real-robot operation.

### 论文解读
#### 摘要翻译
CaSCo（Cascade-Aware Soft-Collision Motion Planning）面向拥挤环境提出软碰撞运动规划：机器人不必把所有接触都视为失败，而是结合物体的语义风险与物理级联后果，寻找总体风险最低的路径。方法使用视觉语言模型评估物体风险，并用物理仿真预测被推动物体是否会继续撞击其他物体。

#### 方法动机分析
传统规划只有“碰撞/不碰撞”两种判断，无法区分擦碰纸箱和碰倒玻璃杯；只检查机器人与物体的直接接触，也可能漏掉“机器人推A、A再撞B”的级联风险。CaSCo 的核心假设是：物体可根据类别和属性获得风险分数，给定动作与当前排列后，仿真器可以近似预测物体后续运动。

#### 方法设计详解
输入是机器人起终配置、环境排列、物体几何和语义属性。视觉语言模型为物体给出0–4级风险。系统在连续配置空间构建概率路图，每条边代表一段机器人轨迹；物理模拟器根据当前排列和轨迹输出新排列，并区分直接受力物体与由物体相互作用产生的级联物体。搜索状态扩展为“机器人位置—物体排列—已支付风险集合”，路径代价为所有被扰动物体风险之和，同一物体重复接触只计一次。A* 使用级联松弛启发式估计剩余代价，并以惰性仿真仅检查有希望的候选边；Pareto 缓存复用启发式结果。执行时若真实排列与预测排列偏差超过阈值δ，则触发重新规划。

#### 方法对比分析
无碰撞规划器在拥挤场景可能无解；静态加权碰撞方法不能更新物体状态；只追踪直接碰撞的方法会漏掉低风险物体撞击贵重物体的后果。CaSCo 的关键区别是把语义风险、动态排列、级联物理和一次性集合代价统一进搜索状态，适合整理、取物和仓储等允许少量可控接触的任务。

#### 实验分析（精简版）
在 MuJoCo Franka 机械臂的货架场景及桌面场景中，CaSCo 的货架中位风险为18.5，低于 Weighted Static MCR 的24.0和 Direct-Only Dynamic 的26.0。相对 Dijkstra，A* 加级联松弛启发式使规划时间缩短66%–79%，物理仿真调用减少74%–92%。结果支持级联建模和惰性仿真的价值，但方法仍依赖仿真准确性，感知不确定性处理有限。

#### 实用指南
复现需要 MuJoCo、Franka 模型、物体碰撞几何、语义风险接口、路图与A*搜索，并记录风险、规划时间和仿真调用数。论文称代码和实物操作数据将在发布时公开，但未给出可核对的仓库地址。迁移到其他机器人时需替换运动学、碰撞和执行器模型，重新校准物理参数、风险分数及偏差阈值δ。

#### 总结
核心思想：让机器人预判碰撞级联风险
1. 用视觉语义区分物体后果。
2. 用物理仿真预测直接与级联扰动。
3. 在增广状态中搜索一次性风险最低路径。
4. 以启发式、惰性仿真和缓存提升效率。
5. 执行偏差超阈值时重新规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.18910v1)
- [arXiv](https://arxiv.org/abs/2609.18910v1)

---

