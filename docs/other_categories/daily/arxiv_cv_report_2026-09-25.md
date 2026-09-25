time: 20260925

# Arxiv Computer Vision Papers - 2026-09-25

## Table of Contents

1. [SARFusion: Scene-Aware Routing Fusion for Robust Camera-LiDAR 3D Object Detection](#2609.29235v1)
2. [FMCW-LIO: A Doppler LiDAR-Inertial Odometry](#2609.29374v1)
3. [CAMP: Cooperative Arm-Hand Motion Planning in Constrained Spaces](#2609.29021v1)
4. [WRAP: Fixtureless Wrench-aware Multi-Robot Assembly Planning](#2609.29407v1)
5. [PolyUMI: Accessible Visual-Tactile-Audio Data Collection for Object Inference and Manipulation](#2609.29760v1)
6. [Echo in the Steps: Learning Perceptive Humanoid Parkour with Gated Memory](#2609.28960v1)
7. [TactileStep: Sole Tactile Learning for Regulating Foot-Terrain Interaction in Humanoid Locomotion](#2609.28959v1)
8. [UCON: Uncertainty-aware Navigation with Historical Re-association in Dynamic Environments](#2609.29419v1)
9. [DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models](#2609.29092v1)
10. [Markerless Multi-Modal Autonomous Robotic Inspection of Large Space Structures](#2609.29644v1)

---

## Papers

<a id='2609.29235v1'></a>
## [SARFusion: Scene-Aware Routing Fusion for Robust Camera-LiDAR 3D Object Detection](https://arxiv.org/abs/2609.29235v1)

**Authors:** Yuting Zhao, Ziyi Zheng, Shuxiao Li

**Published:** 2026-09-24

**Categories:** cs.CV, cs.AI

**Abstract:**

Camera-LiDAR fusion has become a prevailing paradigm for 3D object detection in autonomous driving. However, existing fusion detectors often establish strong inter-modality dependencies by decoding object queries from tightly coupled multimodal representations. Under corrupted driving conditions, such dependencies make the detector vulnerable to unreliable modalities, where degraded observations may interfere with reliable modality-specific evidence and lead to suboptimal predictions. Moreover, modality reliability can vary across both global driving scenes and individual object queries, requiring adaptive fusion decisions at a finer granularity. To bridge this gap, we reformulate robust camera-LiDAR fusion as a scene-aware branch routing problem and propose SARFusion, a robust 3D object detector. Instead of producing detections from a single fused representation, SARFusion decouples object-query decoding into three parallel reasoning branches: a camera branch, a LiDAR branch, and a camera-LiDAR fusion branch. Guided by a Scene Reliability Prior estimated from the global driving context, SARFusion further incorporates object-level evidence to route each query to the most suitable branch. This query-wise routing strategy alleviates harmful cross-modal interference while preserving the benefits of multimodal fusion when complementary cues are trustworthy. On the nuScenes test set, SARFusion achieves strong performance with 72.5 mAP and 74.4 NDS. Extensive analyses demonstrate its robustness under challenging conditions, including sensor corruptions and environmental changes.

### 论文解读

#### 摘要翻译
相机与LiDAR融合检测器依赖紧密耦合的多模态表示；一类传感器受损时，不可靠观测可能连带干扰可靠信息。SARFusion将鲁棒融合改写为场景感知路由：同时保留相机、LiDAR、相机-LiDAR融合分支，根据全局场景条件和单个目标的局部观测质量，为查询选择解码分支。作者报告在nuScenes测试集取得72.5 mAP和74.4 NDS，并称对传感器损坏及环境变化更稳健。

#### 方法动机分析
相机有丰富外观语义、利于远处识别，但深度不确定；LiDAR几何定位准确，却可能因点稀疏而漏掉远处或遮挡目标。固定融合容易让坏模态污染好模态，整帧统一选模态又忽略同一场景内目标间的差异。论文的假设是同时利用帧级天气、光照等可靠性背景和对象级可见度、点密度等局部证据，逐查询选择更合适的信息路径。

#### 方法设计详解
多视角图像和点云分别编码成相机token与LiDAR BEV token，并拼接形成融合上下文。网络保留三个候选Transformer解码分支，分别只读取相机、只读取LiDAR或读取两者。场景先验模块加入可学习场景token，让Transformer汇总全局上下文，经MLP得到可靠性向量；训练时用描述天气和时段的文本提示，通过双向对比损失对齐该向量，推理不需要文本。对每个三维目标查询，把参考点投影到图像特征面和LiDAR BEV面，在两个局部邻域做掩码注意力，汇集局部证据。路由器将局部证据与全局先验拼接，经MLP和Softmax估计三分支概率，再把查询送到概率最大的分支解码检测框。训练还用三分支检测损失和路由交叉熵；模态退化或dropout提供路由监督。作者采用先训练检测分支、再学习场景先验、最后优化路由并联合检测目标的分阶段策略。

#### 方法对比分析
相较固定融合，SARFusion保留互相独立的单模态解码路径，减少坏模态的跨模态干扰；相较整帧切换，它逐对象路由；相较只看局部特征置信度，路由还读取天气、光照等场景先验。其主要贡献是将三条候选路径、全局可靠性和局部查询证据组合起来，不是新的图像或点云编码器。适用于不同目标、不同场景的模态质量变化明显的多传感器检测。

#### 实验分析（精简版）
实验使用nuScenes官方700个训练、150个验证、150个测试场景，以mAP和NDS评估。测试集为72.5/74.4，较CMT的72.0/74.1高0.5/0.3个百分点。验证集雾天SARFusion为69.2 mAP，较CMT的61.4高7.8个百分点；雪、雨、强阳光下分别为64.8、69.7、68.3 mAP，也领先所列对比方法。直接训练与分阶段训练的验证结果分别为65.7/69.6和71.1/73.7。作者称退化评估无需按天气微调，但未说明退化强度和生成细节；模型在雪天仍较干净条件下降6.3 mAP，也没有跨数据集、运行成本或多次运行方差结果。

#### 实用指南
作者表示代码将公开，但论文未给仓库、权重或许可证链接。复现需实现三分支解码、参考点局部掩码注意力、天气/时段提示对比学习和模态退化路由监督，并遵循“检测分支—场景先验—路由”训练阶段；数据划分和mAP/NDS评估应与nuScenes一致。主干、学习率、批量大小、训练轮数、损失权重和硬件均未披露，需自行确定。迁移到其他传感器或机器人时，应替换编码器、局部投影关系及退化监督并重新训练。

#### 总结
核心思想：按场景与目标可靠性路由
1. 编码相机、LiDAR，保留三条候选解码分支。
2. 用场景文本监督全局可靠性先验。
3. 汇集目标查询投影邻域的局部证据。
4. 合并两类证据，为查询选择分支并检测。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29235v1)
- [arXiv](https://arxiv.org/abs/2609.29235v1)

---

<a id='2609.29374v1'></a>
## [FMCW-LIO: A Doppler LiDAR-Inertial Odometry](https://arxiv.org/abs/2609.29374v1)

**Authors:** Mingle Zhao, Jiahao Wang, Tianxiao Gao, Chengzhong Xu, Hui Kong

**Published:** 2026-09-24

**Categories:** cs.RO, cs.CV, eess.SY

**Abstract:**

Conventional LiDAR-inertial odometry (LIO) or simultaneous localization and mapping (SLAM) methods heavily rely on geometric features of environments, as LiDARs primarily provide range measurements instead of motion measurements. From now on, however, the situation changes thanks to the novel Frequency Modulated Continuous Wave (FMCW) Doppler LiDARs. FMCW Doppler LiDARs not only offer the point range with high resolution but also capture the instant point Doppler velocity through the Doppler effect. In the letter, we propose FMCW-LIO, a novel and robust LIO, leveraging intrinsic Doppler measurements from FMCW Doppler LiDARs. To correctly exploit Doppler velocities, a motion compensation method is designed, and a Doppler-aided observation model is applied for on-manifold state estimation. Then, dynamic points can be effectively removed by the Doppler criteria, deriving more consistent geometric observations. FMCW-LIO eventually achieves accurate state estimation and static mapping, even in structure-degenerated environments. Extensive experiments in diverse scenes are performed and FMCW-LIO outperforms other algorithms on both accuracy and robustness.

### 论文解读

#### 摘要翻译
传统激光雷达惯性里程计（LIO）主要依赖环境几何，因为激光雷达通常测距而不直接测运动。调频连续波（FMCW）多普勒激光雷达既提供高分辨率距离，也能测得点的瞬时径向速度。论文提出 FMCW-LIO，借助 IMU 补偿多普勒量测，在流形上进行多普勒辅助状态估计，并据此删除动态点、改善几何观测。作者报告，该方法在结构退化环境中仍能准确估计状态与构建静态地图，在多种场景下精度和鲁棒性优于比较算法。

#### 方法动机分析
隧道等场景的几何约束不足，传统扫描到地图匹配容易漂移；行人和车辆又会污染点云对应关系及地图。论文利用静态点的径向多普勒作为不依赖历史几何匹配的速度线索，先稳定运动估计，再清理动态回波。这个设计依赖可靠的原始多普勒、IMU和静态点，并未消除对几何建图的需求。

#### 方法设计详解
输入是逐点带时间戳的距离与多普勒、IMU角速度/加速度及外参。滤波器维护姿态、位置、速度、IMU偏置、重力和LiDAR-IMU外参；IMU在流形上传播状态与协方差。算法将各采样时刻的点坐标及径向速度补偿到扫描末端，修正不同帧间的距离比例、传感器运动和视线变化，也纳入LiDAR杆臂转动造成的速度。
随后把点的视线方向和多普勒组成线性速度观测，以三点RANSAC抑制动态离群，再用最小二乘拟合三维LiDAR速度；文中示例以30%离群率、99%成功率设定11次迭代，并按视线与速度的夹角加权采样。这个速度先经误差状态滤波更新。再将每点实测多普勒与静态点预测值比较，超过角度相关阈值的点视为动态点而剔除。剩余点形成点到平面残差，交给流形迭代误差状态卡尔曼滤波（IESEKF）完成第二次更新和地图维护。论文使用ikd-Tree局部地图，范围1000米；实验传感器为10 Hz LiDAR、200 Hz IMU。

#### 方法对比分析
FAST-LIO2等方法主要依靠扫描到地图的几何约束。FMCW-LIO在类似的滤波几何更新前加入跨帧多普勒补偿、速度观测和动态点筛除，针对退化与动态污染；不同于只将多普勒放入ICP目标的DICP，它构成IMU融合里程计和静态建图流程。该方法适用于能获得原始Doppler且需要处理弱几何或动态物体的场景。

#### 实验分析（精简版）
实验包括校园、窄街和S形隧道的八段实采序列，比较LINS、LIO-SAM、DLIO、FAST-LIO2及STEAM-DICP，并消融多普勒补偿和动态点移除。最长的1712米隧道序列上，FMCW-LIO达到0.13米RMSE、0.02米终点误差；FAST-LIO2对应为60.09米和23.84米。结构化与退化环境中平均每扫描处理39.71和21.33毫秒，低于10 Hz扫描的100毫秒周期。补偿对快速隧道序列尤其重要；动态移除改善拥挤街道结果，但并非所有场景和指标均提升。地图去除动态物体的展示为定性结果，论文未报告检测精确率或召回率。

#### 实用指南
论文说明FMCW-LIO数据集（五段手持序列）和Free-Init数据集（四段动态初始化序列）已发布，但所读文本未给出项目链接；未明确说明算法代码开源。复现需原始逐点多普勒、点级时间戳、IMU同步与外参标定，并实现速度补偿、RANSAC拟合、动态点判定和两阶段滤波。实验用Intel i7-1165G7、ikd-Tree和1000米局部地图；完整噪声、标定与筛点阈值未列全。迁移到其他传感器时应重新标定并验证时序、多普勒噪声及运动条件。

#### 总结
核心思想：用多普勒速度补几何退化

1. IMU补偿逐点距离与多普勒。
2. RANSAC从径向速度估计LiDAR速度。
3. 用速度预测并剔除动态点。
4. 对保留点进行滤波更新和静态建图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29374v1)
- [arXiv](https://arxiv.org/abs/2609.29374v1)

---

<a id='2609.29021v1'></a>
## [CAMP: Cooperative Arm-Hand Motion Planning in Constrained Spaces](https://arxiv.org/abs/2609.29021v1)

**Authors:** Ziyuan Wang, Yunlong Shan, Fei Mo, Sichao Liu, David Navarro-Alarcon, Jia Pan, Kosta Jovanovic, Xin Jiang, Peng Zhou

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

Coordinated arm-hand motion planning is fundamental to dexterous robotic manipulation in complex and constrained environments. A straightforward solution is to decompose the problem into separate arm path planning and hand motion generation; however, this poses a dilemma: decomposition can miss feasible solutions that require coordinated arm-hand adaptation along the path. Alternatively, directly planning in the high-dimensional joint arm-hand configuration space captures such coupling but faces a substantially enlarged search space and nonconvex collision constraints. To characterize this coupling, we formulate feasible hand fibers that capture collision-free hand configurations for each arm configuration. Based on this formulation, we propose CAMP, a high-success and efficient cooperative arm-hand motion planner for constrained environments. CAMP constructs candidate trajectories through layered hand search with local arm relaxation, then compactly represents them using endpoint-preserving via-point movement primitives (VMPs) for coarse-to-fine joint optimization. Across six constrained simulation tasks, CAMP achieves 84.2-98.5% planning success, outperforming alternative planners with competitive efficiency. Ablation studies verify the contributions of arm relaxation, VMP representation, and coarse-to-fine optimization, while real-robot experiments demonstrate CAMP on constrained manipulation tasks. The project website is available at https://camp-armhand.github.io/.

### 论文解读

#### 摘要翻译
受限环境中的灵巧操作要求机械臂与手协同运动。分开规划容易漏掉必须沿程改变臂手姿态的解，直接在高维联合空间搜索又很困难。论文提出 CAMP，以“可行手部纤维”描述每个臂姿态下无碰撞的手形集合，再结合分层手部搜索、局部臂路径松弛和保持端点的 VMP 轨迹优化。六项仿真任务成功率为 84.2%–98.5%，并进行了真实机器人验证。

#### 方法动机分析
臂移动会改变周围障碍对手形的限制；即使一条臂路径处处存在可行手形，也不代表这些手形能连续地从起点连接到终点。固定手形会限制可走的臂路径，固定臂路径又可能阻断手形变化。CAMP 的关键假设是用低维臂路径提供全局方向，同时允许局部调整臂和手，以处理两者互相影响的碰撞约束。

#### 方法设计详解
先用 RRT-Connect 在臂关节空间生成多条路线，并按进度分层。每条路线上的双向 Hand-RRT 从指定起始手形和目标手形出发，在各层搜索无碰撞手形并尝试连接相邻层。算法检查层间中点；若插值碰撞，就在该小段两端及中点共同调整手关节和有界臂关节偏移，再拼回原路线。这样既保留全局路线，又能在手部可行集合断开的地方改变机械臂。

每个候选臂手轨迹随后由端点保持 VMP 表示：线性端点参考叠加高斯基函数加权形变，端点包络在起终点归零，因此优化权重不会移动任务端点。初始化通过轨迹拟合、权重正则和二阶平滑共同求解。粗阶段并行筛选多条候选，较强地约束其偏离初始臂路径，优先消除碰撞和其他约束违反，但臂与手都可以调整；选出前三条后，细阶段放松路径锚定并增加轨迹质量权重。目标兼顾关节平滑、偏离初始轨迹和距离场碰撞惩罚，同时受关节位置、速度、环境碰撞、自碰撞及逐点手形可行性约束。

#### 方法对比分析
相较于固定开手或紧凑手的分阶段方案，CAMP 可沿路径改变手形；相较于全 22 自由度直接搜索，它先利用低维臂空间生成路线；相较于固定路线提升和单一初始化优化，它能局部移动臂路径并优化多种候选。主要创新贡献在于局部臂松弛与手形搜索形成闭环，再用多候选 VMP 优化提升可行性；相比单独更换某个标准规划器，区别是显式处理“臂姿态改变可行手形集合”的双向耦合。代价是需要准确的机器人/障碍碰撞模型和较多候选优化，当前重点仍是已知静态环境。

#### 实验分析（精简版）
仿真平台为 6 自由度 UR7e 加 16 自由度 LinkerHand；规划使用 PyRoKi 距离场，MuJoCo 作精确碰撞验证。每任务每方法进行 10 批、每批 100 次。CAMP 在六项任务均获最高成功率：墙面穿越 98.5%、多球避障 91.3%、窄通道 92.5%、盒中球预抓取 92.0%、按键接近 94.4%、柜内圆柱预抓取 84.2%。例如窄通道成功率高于 A*+CHOMP 的 71.5%，时间为 30.10 秒，对方为 53.80 秒。盒中球消融中，固定臂成功率 74%，加入臂松弛后为 92%；VMP 相比逐路点优化为 92% 对 79%，平均时间 32.45 对 84.97 秒。真实机器人每任务 10 次，成功率为 80%、90%、80%；其中按键任务只验证指尖到达目标，并未实际按键，预抓取任务也不要求真实抓取或抬起。失败主要发生在窄间隙，反映定位、标定与执行误差影响。

#### 实用指南
论文给出 UR7e+LinkerHand、i9-14900KF/RTX 5070 Ti 的仿真配置；每轨迹 200 路点，初始化 16 路并行实例、20 层，局部臂偏移 ±0.1 rad；VMP 每臂关节 30 个基函数、每手关节 20 个，粗/细阶段迭代上限 100/200，保留 3 条候选精修。实机按已知障碍模型规划并留 1 cm 碰撞裕量。论文提供项目网站 camp-armhand.github.io，但未明确交代代码、模型或数据是否开放；迁移时需替换运动学、碰撞几何和关节限制，并重新验证。

#### 总结
核心思想：臂手协同穿越可行纤维

1. 生成多条臂空间路线。
2. 分层搜索手形，碰撞处松弛臂路径。
3. 以端点保持 VMP 压缩候选。
4. 粗筛可行性、细调轨迹质量。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29021v1)
- [arXiv](https://arxiv.org/abs/2609.29021v1)

---

<a id='2609.29407v1'></a>
## [WRAP: Fixtureless Wrench-aware Multi-Robot Assembly Planning](https://arxiv.org/abs/2609.29407v1)

**Authors:** Valentin N. Hartmann, Huang Su, Yijiang Huang, Stelian Coros

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

Assembly using robots often requires specially designed fixtures, or relies on top-down only assembly strategies. Using multiple robots, we can avoid using fixtures and make robotic assembly more flexible. Planning assembly sequences for multiple robots is challenging due to the high number of possible task assignments and orders. In addition, we need to reason over forces that occur during the assembly process, e.g., to decide if multiple robots are required for support, or if external support such as a table should be used.   We present Wrap, a multi-robot assembly planner for multi-part assemblies, given the inter-part ordering-dependencies, the part meshes, and their initial state. We formulate a linear program to reason about valid grasps for supporting the forces that occur during assembly. The search leverages the assembly sequence, and greedily finds a feasible solution per assembly step by computing a heuristic via a cheap backwards search, and using the heuristic in the more expensive forward search.   We then solve the multi-robot, multi-goal motion planning problem, and for execution, we split the plan into contact-rich assembly skills, and free space motion. We benchmark the planner on a variety of multi-part assemblies, and apply the planner to groups of robots differing in size and kinematics. We validate the work both in a physics simulation, and in real. Videos and code are available at https://www.vhartmann.com/wrap.

### 论文解读

#### 摘要翻译
机器人装配往往需要定制夹具，或受限于自上而下的顺序流程。WRAP 面向多零件、多机器人任务，输入零件间依赖关系、网格与初始状态，用线性规划判断抓取能否承受装配力；廉价反向搜索为正向搜索提供启发式，逐步求出可行装配计划，再生成多机器人运动并拆分为接触装配技能和自由空间运动。作者在仿真与真实机器人上验证了方法。

#### 方法动机分析
多机器人可替代夹具、提升装配灵活性，但任务分配、操作顺序与抓取选择组合巨大；压配等任务还会产生侧向力和倾覆力矩。只检查几何可达性，可能选出无法承载配合力的抓取。WRAP 的关键假设是零件及装配依赖已知，可用机器人、桌面与零件接口的扳手能力描述支撑；目标是在搜索成本可控的同时避免不可执行的受力方案。

#### 方法设计详解
输入为零件网格与初始位姿、装配依赖和目标相对位姿，并描述机器人末端、桌面及接口扳手能力。首先为子装配采样稳定放置姿态和抓取。对每个零件建立静力平衡：机器人扳手、桌面支撑、相邻零件接口力与重力等外力之和为零，再以线性规划检查各扳手是否落在能力集合内。鲁棒检查还要求有界插入力不确定集的每个顶点均可平衡。搜索按装配里程碑展开；反向 Dijkstra 在抽象状态上用碰撞与力约束生成启发式，正向惰性搜索再逐个动作做逆运动学验证，并缓存成功构型、挂起暂时失败项以增加后续搜索预算。大邻域搜索扩展每次填充的动作窗口，窗口内结合贪心搜索和有上界的 A*；随后优化关键帧关节运动与机器人间净空，规划自由空间轨迹，并用局部交互力控制执行接触密集的插接。该规划假设为抓取式操作，不以刚性接口替代其受力能力模型。

#### 方法对比分析
传统装配序列规划多以零件为中心，机器人仅作可行性筛选；WRAP 把抓取、机器人分配和受力支撑共同纳入搜索。与 AutoMate 的接触策略学习不同，它面向多零件次序和多机器人任务；与 Fabrica 相比，它显式检查配合扳手并可重新抓取或调整姿态，而非固定部分装配体位姿、只用抓取稳定性启发式。代价是依赖已知依赖和扳手模型，以及昂贵的碰撞与 IK 检查。

#### 实验分析（精简版）
仿真使用 MuJoCo，默认插入力 10 N、未另行说明时不加噪声，通常每项重复 10 次。Cross、Cube、Chair 三种任务中，不做受力检查或仅考虑重力时成功率均为 0%；完整模型分别为 100%、90%、100%，Cube 有一次插接就位失败。结果说明桌面和多机器人支撑约束对力控装配有实际作用，但并不代表所有零件与摩擦条件都能成功。实机由固定在 Husky 底座上的两台 UR5e 完成半张凳子装配，计划含 7 个动作、一次交接和两次插腿。论文未报告多次实机成功率；模型也忽略了零件受载弯曲及内部应力。

#### 实用指南
常规规划推理每个末端采样20个抓取；Fabrica用64个固定抓取且关闭抓取扩展。实机夹爪限值经施力至滑移/释放测试标定，名义规划力设为 25 N（含安全系数）；论文指出仿真并未精确复现真实接触摩擦。复现需实现碰撞检查、优化式逆运动学、力可行性线性规划、多目标运动规划与接触控制。摘要给出项目主页作为代码和视频入口，但未详述许可、模型权重及数据集发布。这是显式搜索规划器；迁移时需校准运动学、抓取与接口扳手集合及装配依赖。

#### 总结
核心思想：用力可行性驱动多机器人装配
1. 按依赖图生成装配里程碑和可用抓取。
2. 用静力平衡 LP 筛除无法支撑插入力的方案。
3. 以反向启发式引导正向 IK 搜索并优化操作序列。
4. 联合运动规划，把接触片段交由力控技能执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29407v1)
- [arXiv](https://arxiv.org/abs/2609.29407v1)

---

<a id='2609.29760v1'></a>
## [PolyUMI: Accessible Visual-Tactile-Audio Data Collection for Object Inference and Manipulation](https://arxiv.org/abs/2609.29760v1)

**Authors:** Conor W. Hayes, Rickmer Krohn, Aravind Ramaswami, Anunth Ramaswami, Nils Dengler, Kevin M. Lynch, J. Edward Colgate, Georgia Chalvatzaki, Matthew L. Elwin

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

Humans typically rely on vision, touch, hearing, and proprioception to perceive contact and adapt their actions during manipulation. Providing robots with comparable responsiveness therefore requires hardware that can retain and use these complementary sensory signals. Most imitation-learning systems, however, observe demonstrations primarily through vision and proprioception, limiting access to contact information that is difficult to infer visually. We present PolyUMI, an open-source platform for scalable visual--tactile--audio demonstration collection and robot deployment. Its lightweight, wireless handheld gripper records synchronized wrist-camera, optical tactile, contact-audio, and proprioceptive observations without requiring a tethered workstation. The same sensing finger can be transferred to the robot end effector, preserving the sensing geometry between demonstration collection and policy execution. To effectively use these heterogeneous observations, we further introduce VisTA, a token-level multimodal policy that integrates information across sensors and time to predict contact-aware robot actions. Experiments spanning object inference, slip control, and contact-rich manipulation show that touch and audio reveal task-relevant information beyond vision and that VisTA is competitive with or outperforms existing multimodal policies. Together, PolyUMI and VisTA provide an accessible pipeline for collecting multimodal demonstrations and learning policies that perceive physical interaction beyond vision. Project Page: https://polyumi-vista.github.io

### 论文解读

#### 摘要翻译
机器人模仿学习常只记录视觉与本体状态，难利用视觉难以推断的接触信息。本文提出开源平台 PolyUMI：无线手持夹爪同步记录腕部相机、光学触觉、接触音频和本体感觉；同一传感手指可装到机器人末端，保持采集与执行的感知几何一致。作者还提出 VisTA，以 token 级融合传感器与时间信息，预测接触感知动作。实验显示，触觉、音频可补充视觉，VisTA 在部分任务优于现有多模态策略。

#### 方法动机分析
视觉难以揭示遮挡物体属性、细小纹理、滑移与真实接触状态，模仿会缺失反馈。作者假设光学触觉表征接触形变，接触音频记录碰撞或摩擦，二者与视觉、本体状态互补。另一个痛点是示范器与机器人传感布局不同会造成观测偏移；同一传感手指用于缩小差异。若任务状态清晰可见，多模态未必胜过视觉。

#### 方法设计详解
PolyUMI 有手持夹爪与机器人安装形态，共用同一传感手指。触觉相机透过弯曲镜观察七层 VHB 胶带和铝粉反光层的形变，以 20 fps 采集；接触麦克风以 16 kHz 录制结构传导声音；腕部 GoPro 约有 177° 视场、60 fps，另记录本体状态。传感流按 10 Hz 时间轴对齐到最高延迟流的最新时间戳，并校正执行延迟。视觉和触觉图像缩至 224×224，使用当前及前一帧；音频转为 128 Mel 频带、48 帧（约 0.5 秒）。输入图像由 CNN 各编码 98 个 token，音频编码 96 个，状态由 MLP 编码，宽度均为 624。八层、八头 transformer 对共 294 个 token 做跨模态和跨时间双向注意力，使触觉与声音事件关联视觉上下文。融合结果通过交叉注意力条件化 18 层 DiT 动作模型，以条件流匹配预测未来 16 步、每步 10 维的相对末端位姿及夹爪宽度。控制采用滚动规划：时间集成后先执行前三步，再重新观测预测。策略在 Franka FR3 上训练 120 epochs；论文未给学习率和批量大小。

#### 方法对比分析
PolyUMI 的核心贡献是无线同步采集视觉、触觉、音频，并让同一传感手指从演示装置迁至机器人。VisTA 保留局部 token 并统一时空融合；MulSA、Sparsh-X、PolyTouch 则采用独立编码、注意力瓶颈/池化或预训练表示。VisTA 适合需持续接触、视觉难判断接触质量的操作；视觉线索充分时，多模态可能无益。

#### 实验分析（精简版）
触觉形状识别在场景隔离测试集准确率为 92.3%。盒中物体分类每类用 20 段训练示范，无音频配置为 42%–44%，加音频约 80%，随机猜测为 33%；触觉＋音频和三模态对粗螺丝达 100%。滑移控制每种配置测 10 次，三模态成功 8/10（80%），纯视觉 2/10（20%），纯音频 0/10。擦拭中 VisTA 更能保持接触并擦完线，但正文未列精确成功率；灯泡旋紧各策略至少 80%，纯视觉最佳，VisTA 与最强多模态基线相当。价值依任务而异；操作仅测两个任务、单一机器人，滑移样本少。

#### 实用指南
论文称 PolyUMI 开源，项目页 https://polyumi-vista.github.io 介绍硬件、电子、固件、制作说明和学习软件；未明确数据集链接或许可证。复现需保持手指几何一致，处理 10 Hz 时间对齐及观测延迟，并按文中采样率预处理。训练 120 epochs；学习率、batch size、优化器未说明。迁移到其他机械臂需适配安装件、状态和控制器并重新采集演示，跨机器人泛化尚无证据。

#### 总结
核心思想：让接触信号进入策略
1. 用可转移传感手指采集同步多模态示范。
2. 校正异步视觉、触觉和音频的时间差。
3. 将各传感器编码为时空 token 并逐 token 融合。
4. 以融合结果驱动流匹配策略预测动作块。
5. 执行部分动作后重新感知，闭环调整抓持与接触。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29760v1)
- [arXiv](https://arxiv.org/abs/2609.29760v1)

---

<a id='2609.28960v1'></a>
## [Echo in the Steps: Learning Perceptive Humanoid Parkour with Gated Memory](https://arxiv.org/abs/2609.28960v1)

**Authors:** Ming-Ju Lee, Zizhuo Wang, Shaoting Zhu, Haozhe Lou, Hang Zhao, Yiming Li

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

While recent advances in perceptive locomotion have enabled humanoid robots to traverse structured terrains, agile parkour in highly discontinuous environments remains an open challenge. In particular, crossing sparse footholds and narrow support regions requires precise foothold selection, effective use of visual observations, and consistent alternating foot placement during fast transitions. In this paper, we present a perceptive humanoid parkour framework that enables stable traversal across terrains with limited foothold availability using only onboard depth observations. The framework features a saliency-guided temporal perception module that combines a saliency prior with gated memory. It retains informative depth features across frames, enabling reliable foot placement from partial observations. By introducing an alternation loss, our symmetry regularization encourages alternating gait patterns and improves traversal robustness. Extensive experiments show that our method significantly improves success rate and foothold accuracy on challenging terrains in both simulation and the real world.

### 论文解读

#### 摘要翻译
感知运动控制虽已帮助人形机器人穿越结构化地形，但在高度不连续环境中敏捷跑酷仍很困难。跨越稀疏落脚点和狭窄支撑面，要求精确选点、有效利用视觉并在快速转换时保持双脚交替。本文提出仅依靠机载深度感知的跑酷框架，以显著性先验和门控记忆跨帧保留有用深度线索，并以交替损失改善步态。仿真和实机实验显示，该方法提高了复杂地形上的成功率与落脚准确性。

#### 方法动机分析
单帧深度可能看不到完整落脚几何；地图方法又依赖定位和稳定更新，在动态跑酷中容易受限。历史视角能补充被遮挡或已离开视野的线索，但直接累积会引入冗余。镜像一致性虽能利用人形双侧结构，却可能让左右腿同步跳跃，不适合窄支撑上的连续换脚。作者以“显著性筛选有价值历史、门控融合补充几何、交替约束组织步态”为核心假设。

#### 方法设计详解
深度图先计算垂直相邻像素差，并按图像高度加权，突出近身几何边缘；像素显著性均值形成帧级先验。共享编码器处理当前帧与历史帧，MLP结合历史潜变量和相对显著性预测softmax权重。门控模块将加权历史特征相对当前潜变量的残差注入当前表示，避免简单平均。融合视觉与本体感觉历史后，由MLP策略输出29个关节目标位置；PPO非对称actor–critic训练时，critic还使用特权状态。镜像一致性约束双侧动作，交替损失只在机器人移动时惩罚两腿动作同向。训练在Isaac Sim/Isaac Lab中使用2048个并行G1；本体感觉历史为8帧、深度历史为4帧，地形用20×10课程网格逐步增加难度。

#### 方法对比分析
与高度图或体素地图不同，该方法不依赖全局重建；与直接深度策略不同，它显式强调支撑几何并利用可学习记忆。关键创新是显著性权重与门控残差的组合，以及在镜像正则中加入交替步态约束。它面向稀疏落脚、狭窄支撑的快速移动；错列落点的支撑腿规划和急转时的视野不足仍未解决。

#### 实验分析（精简版）
仿真比较覆盖箱、桩、梁、楔和梯形地形，每实验5000次、三个随机种子。相较同设置训练的Hiking基线，主结果平均成功率为92.7%、落脚准确率83.6%，分别提升16.7和7.8个百分点；梯形地形成功率为97.1%，基线为64.2%。消融中，学习权重加显著性后成功率/准确率为88.14%/90.63%，再加门控记忆为98.27%/91.10%；完整模型消融汇总与主结果平均不一致，论文未解释。对称正则在箱地形将成功率从85.0%提高至96.7%。实机每类地形测试10次，但只报告高成功率，未公布精确数值或不确定性；因此实机证据主要是定性验证。

#### 实用指南
论文提供项目主页 https://echo-in-the-steps.github.io/，但未明确说明代码、模型权重或数据开放。复现需实现深度预处理、4帧记忆、显著性调权、PPO非对称critic及双腿镜像映射；已报告初始学习率为1×10⁻³，使用RTX 4090训练。实机使用RealSense D435i（60 Hz）和Orin NX，深度由480×270缩至64×36后裁为32×18，策略以50 Hz推理。相机标定、部分归一化和延迟设置未完整交代。迁移时需重做相机/关节映射并针对新机器人与地形重新训练；作者建议以脚步规划和广角或主动感知处理交叉步及急转视野不足。

#### 总结
核心思想：显著性记忆辅助交替落脚
1. 用深度跳变定位重要支撑线索。
2. 按显著性加权历史帧并门控融合残差。
3. 融合视觉与本体感觉，预测关节目标。
4. 加入交替损失，减少同步跳跃并稳定换脚。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.28960v1)
- [arXiv](https://arxiv.org/abs/2609.28960v1)

---

<a id='2609.28959v1'></a>
## [TactileStep: Sole Tactile Learning for Regulating Foot-Terrain Interaction in Humanoid Locomotion](https://arxiv.org/abs/2609.28959v1)

**Authors:** Zizhuo Wang, Ming-ju Lee, Shaoting Zhu, Haozhe Lou, Hang Zhao, Yiming Li

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

Humanoid parkour policies can traverse various terrains, but task completion may mask challenges of harsh landings, edge contacts, and unstable stance contacts. Humans naturally regulate foot-terrain interaction through tactile feedback, modulating contact compliance according to terrain stiffness. This highlights a key domain gap between humans and humanoid robots: the absence of rich tactile sensing in most humanoid systems. We address this problem with TactileStep, a deployable tactile learning framework that brings sole pressure sensing into humanoid locomotion control for softer touchdowns and more stable support. TactileStep aligns tactile simulation with the real pressure insole, allowing the policy to learn from the same contact features available on hardware. During training, we use tactile and motion cues to recognize different foot-contact phases and apply phase-aware rewards that encourage safer landing and more stable stance. Evaluated in simulation and on a Unitree G1 humanoid across diverse terrains, TactileStep reduces peak touchdown force by up to 48.8% and peak A-weighted impact noise by up to 30.1 dB over a strong perceptive baseline, while increasing stance contact area by up to 23.8%.

### 论文解读

#### 摘要翻译
TactileStep 面向人形机器人步行，使用可部署的足底压力传感，并在训练和真实控制中持续输入触觉，以同时改善触地柔和度和站立稳定性。作者认为，越过障碍不代表落脚轻柔或支撑充分。仿真及多地形实测显示，相对感知式基线，峰值触地力最多降低48.8%，峰值A计权撞击噪声最多降低30.1 dB，站立接触面积最多增加23.8%。

#### 方法动机分析
视觉能在触地前描述地形，却看不到落脚后压力分布；本体感觉或估计地面反力也难直接表征接触面积、压力中心及脚底是否偏载。论文的核心假设是把鞋底压力摘要纳入部署策略，并按预着地、着地、支撑阶段分别优化，便能把“怎样接触地面”变成可控目标，而不仅追求穿越成功。方法仍依赖刚性接触近似与手工相位规则。

#### 方法设计详解
仿真为每只脚布置60个虚拟压力单元：射线检测鞋底与地形的间隙和法向，按接触方向及距离分配合力，再对相邻单元平滑载荷；随后提取归一化法向力、受力单元比例作为接触面积、以及力加权压力中心CoP。真实鞋垫读数通过校准映射为力。策略输入包括关节与基座状态历史、深度历史，以及左右脚两帧触觉特征；Actor输出29个关节目标位置，由PD控制器执行。每脚根据压力是否超过阈值、足部下行速度和离地高度，在线划分摆动、预着地、着地、支撑四相。奖励在预着地抑制向下速度和加速度，在着地阶段惩罚力及其突增，并针对窗口峰值施加事件奖励；支撑时鼓励更大接触面积和CoP边界余量、抑制CoP抖动。PPO配合双critic分别估计连续运动奖励与稀疏接触/事件奖励，再混合优势更新策略。

#### 方法对比分析
相较Hiking in the Wild视觉感知式parkour基线，本文不只增加一个输入，而是在部署时保留鞋底压力闭环，并将落脚品质直接写入分相奖励；作者也将其与仅把触觉用于训练监督的思路区分。无触觉观测、无软着地奖励、无稳定支撑奖励和单critic消融分别检验传感反馈、着地目标、支撑目标和价值分解。它适合配备可校准足底压力传感的类人机器人；对软地面和高速动作的迁移尚未验证。

#### 实验分析（精简版）
仿真每种策略、每类地形评估4096回合；真实机器人每个条件采集20个样本。平台上台阶的平均触地力从基线695.0 N降至355.7 N（48.8%）；下楼峰值噪声从97.2 dB降至67.1 dB（30.1 dB）；下楼接触面积比从0.483升至0.598（相对增加23.8%）。模拟楼梯下行中，TactileStep接触面积比0.545，高于无稳定奖励的0.455；CoP余量29.43 mm，高于基线27.07 mm。移除软着地奖励会增加冲击。常规行走成功率与基线相当或略高，但速度误差和能耗通常略增；论文未报告显著性检验，且“±”的统计定义未说明。

#### 实用指南
训练使用Isaac Sim、2048个并行环境、50,000次迭代，每次每环境采集24步；策略为PPO，学习率1×10⁻³。部署无线鞋垫采样率25 Hz，采集后的处理与通信延迟小于1 ms；100 Hz有线测试用于检查峰值采样误差。文中提供项目网站，但未明确说明代码、权重或数据是否开放。复现或迁移时需按新鞋垫重新校准压力映射、taxel布局与噪声，并在目标机器人及地面重新评测。

#### 总结
核心思想：足底触觉闭环调节落足接触
1. 由模拟压力单元提取法向力、接触面积与CoP。
2. 用足速、高度和压力划分四相，触发对应接触奖励。
3. 双critic分开估计运动与接触目标，混合优势更新策略。
4. 把关节目标交给PD执行，并根据鞋底反馈调整落足。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.28959v1)
- [arXiv](https://arxiv.org/abs/2609.28959v1)

---

<a id='2609.29419v1'></a>
## [UCON: Uncertainty-aware Navigation with Historical Re-association in Dynamic Environments](https://arxiv.org/abs/2609.29419v1)

**Authors:** Bing Sun, Yue Lin, Yongsheng Yuan, Yang Liu, Dong Wang, Huchuan Lu

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

Autonomous navigation in dynamic environments is hindered by two fundamental challenges: perception instability and uncertainty-optimization mismatch. The former leads to identity switches and unreliable motion estimation, while the latter prevents principled incorporation of motion uncertainty into trajectory optimization. To address these challenges, we propose UCON, an uncertainty-aware navigation algorithm in dynamic environments. For perception instability, we present a point-level historical re-association mechanism that leverages historical point cloud fragments to recover lost targets while maintaining identity continuity. Subsequently, a Kalman filter is employed to provide anisotropic motion state estimation and covariance propagation. To resolve the uncertainty-optimization mismatch, we transform predicted states and their covariances into uncertainty sectors, which are embedded as differentiable cost terms within a trajectory optimization framework. This achieves consistent uncertainty-aware dynamic obstacle avoidance while maintaining smoothness and feasibility. Extensive simulations and real-world experiments demonstrate that, while maintaining high computational efficiency, UCON achieves superior perception stability and robust navigation performance in dynamic environments compared to state-of-the-art methods. The code will be open-sourced to facilitate further research.

### 论文解读

#### 摘要翻译
动态环境导航既受感知不稳定影响，也面临运动不确定性难以进入轨迹优化的问题。UCON用历史点云片段进行点级重关联以恢复目标身份，以卡尔曼滤波估计运动状态和协方差，再把预测状态转为不确定性扇区并加入可微规划代价。作者通过仿真和真实机器人实验报告了感知稳定性与动态避障表现，并称计算效率较高。

#### 方法动机分析
遮挡、目标交叠和稀疏LiDAR观测会导致物体级关联断裂，单帧几何难以找回丢失目标。规划器若只用障碍物均值轨迹，或统一扩大安全半径，也无法区分沿运动方向和横向的风险。UCON的假设是：短时历史点云仍能通过运动预测在当前扫描中找回；协方差则可转成有方向性的安全区域，直接影响轨迹搜索。

#### 方法设计详解
流程从LiDAR点云聚类与运动过滤开始，以水平投影框IoU、速度方向相似性和尺寸变化进行物体级匹配。若匹配失败，方法保留历史物体点云，将各点按预测速度和观测间隔外推，再在当前环境点云的KD树中查找邻点；距离小于0.6 m且匹配数超过原点数35%时恢复点云与身份。随后以位置、速度组成六维恒速状态，通过卡尔曼滤波传播协方差。障碍物风险扇区沿速度方向扩张半径，横向不确定性则增大扇区角度。机器人相对位置的径向与角向余量经两个Sigmoid形成可微动态避障代价。最后，MINCO分段轨迹同时优化该代价、三阶导数平方和行驶时间，并约束静态间距、速度、加速度及段间连续性；数值积分后用L-BFGS求解。

#### 方法对比分析
相对FAPP，UCON新增点级历史恢复，并以方向相关扇区替代椭圆式不确定区域，目标是在保留主要运动方向安全性的同时释放侧向空间。相比Intent-MPC等视觉或学习式方法，它以显式卡尔曼协方差和几何优化为核心，不依赖语义推理。因而更适合关注低延迟和身份连续性的LiDAR导航；语义理解及行人意图建模仍不在其范围内。

#### 实验分析（精简版）
真实感知测试包含三名随机运动行人，每次持续243秒。UCON的MOTA为85.2%、身份切换3次，优于FAPP的70.3%与16次、Intent-MPC的66.4%与17次；平均感知耗时17.78 ms/帧。仿真在50、80、110个动态障碍下各重复50次，完整UCON成功率依次为90%、86%、74%，高于FAPP的88%、78%、66%；110个障碍时不建模运动不确定性的消融版本仅60%。Jetson AGX Orin上规划耗时为2.24–3.02 ms。真实机器人展示了多人阻挡时的避障，但没有给出重复试验统计；因此定量优势主要来自感知测试和仿真。

#### 实用指南
硬件设定为Mecanum全向底盘、Livox Mid-360和Jetson AGX Orin；雷达以50 Hz发布点云，平均每帧约4922点，定位使用Faster-LIO且无需预建地图。复现时应校准运动噪声和测量噪声，并针对点云稀疏度调整0.6 m匹配阈值、35%点数门槛及扇区参数。实验还需报告成功率、规划时间与轨迹平滑度。作者表示未来开放代码，当前论文未提供可确认的代码仓库或公开数据集；迁移到其他传感器和底盘需重新标定并验证安全边界。

#### 总结
核心思想：历史找回身份，协方差塑造避障
1. 聚类LiDAR动态目标并完成几何运动匹配。
2. 用速度外推历史点，在当前扫描中重建失配目标。
3. 将卡尔曼位置协方差转为纵横向不等的风险扇区。
4. 把扇区代价并入MINCO轨迹优化，兼顾避碰与平滑。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29419v1)
- [arXiv](https://arxiv.org/abs/2609.29419v1)

---

<a id='2609.29092v1'></a>
## [DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models](https://arxiv.org/abs/2609.29092v1)

**Authors:** Yohan Choi, Min-Jun Kim, Jin-Sung Kim, Yong-Jae Kim, Youn-Hee Han

**Published:** 2026-09-24

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG

**Abstract:**

Vision-based legged locomotion methods assume clean depth at training time and rely on hand-tuned post-processing filters at deployment. However, filter parameters are rarely disclosed, hindering reproducibility, and performance degrades substantially when depth noise is left unaddressed. Building noise robustness directly into the learning pipeline would eliminate this dependency. While such robustness has been explored for proprioceptive inputs, analogous approaches for depth perception remain largely absent in legged locomotion. We propose DAWN (Denoising and Alignment in World models for Noise-robustness), a noise-robust perception framework for legged locomotion, which builds noise robustness directly into a world model via two modifications: (1) feeding noisy depth to the encoder while keeping clean depth as the reconstruction target, forcing the model to implicitly denoise its input; and (2) applying contrastive learning to align the latent states of noisy and clean depth. Importantly, DAWN is not tied to a specific noise model, requiring no manual tuning to the noise distribution at deployment. Furthermore, it incurs no additional inference cost over existing world model-based methods. Without any manual filter calibration -- relying solely on the learned noise-robust representation -- DAWN achieves zero-shot quadruped parkour on a Unitree Go1: traversing stairs up to 18 cm, clearing gaps up to 70 cm, and mounting steps up to 45 cm from raw depth observations. Ablation studies show that denoising and contrastive alignment contribute at complementary levels -- reconstruction and representation, respectively -- and yield additive gains when combined. Videos and code are available at: https://dawn-parkour.github.io/

### 论文解读
#### 摘要翻译
视觉足式运动通常在干净深度上训练，部署时再用人工调参的滤波器；参数不公开影响复现，未处理噪声则会损害性能。DAWN 将去噪和对比对齐加入世界模型：以带噪深度为输入、干净深度为重建目标，并拉近两者的潜在状态。作者称其无需按部署噪声手调滤波器且不增加推理开销；Go1 实机可通过最高18厘米楼梯、70厘米沟隙和45厘米台阶。

#### 方法动机分析
深度噪声常集中在物体边缘且随距离变化，会破坏楼梯、沟隙等关键几何。仅在训练时加入噪声未必能学到稳定表征，外挂滤波器又需要人工校准。DAWN 假设，以受噪输入预测干净观测可压制无关噪声，再把同一场景的干净/带噪表示显式对齐，能让控制策略保留地形信息。

#### 方法设计详解
方法沿用 RSSM 世界模型和 PPO 控制器。输入由深度图与本体感知组成；GRU 汇总历史状态和动作，编码器得到随机状态，动力学模块预测先验，解码器重建观测，确定性状态供策略输出关节位置目标。DAWN 对同一时刻构造干净和带噪深度配对，带噪观测进入编码器，干净观测作为解码目标；损失包含重建项和后验—先验 KL 正则。另将两种状态送入 MLP 投影头，以余弦相似度计算 NT-Xent：同场景为正对、其他场景为负对。总损失为去噪目标加权叠加对比项。投影头只在训练时使用，部署沿用原世界模型推理路径。训练控制频率为50 Hz、深度分辨率64×64，每5个控制步更新深度；噪声包括标准差0.01米的高斯扰动、边缘丢失和远距粒子噪声。

#### 方法对比分析
WMP 用干净输入重建自身；DAWN 改为带噪输入、干净目标，并增加潜在对齐。它不是另加推理期滤波器，也不改 RSSM/PPO 主干。相较仅做噪声训练，两个目标直接约束几何恢复与表征一致性，适用于深度噪声会影响视觉控制的场景。

#### 实验分析（精简版）
仿真在坡面、楼梯、沟隙和台阶上比较 WMP、噪声训练 WMP、DAWN、Oracle 等方法，使用3个随机种子、每条件100回合。楼梯/沟隙/台阶成功率分别为96.6%、97.2%、97.0%，均值96.9%；干净深度 WMP 对应89.7%、88.6%、95.9%。最高难度平均成功率为88.2%。噪声增至训练尺度的2倍时，DAWN 成功率下降6.5个百分点，干净训练 WMP 下降17.5个百分点。Go1 室外最难楼梯18厘米、沟隙70厘米、台阶45厘米成功率为80%、60%、70%，每难度仅10次；结果有提升但仍会失败，且未报告显著性检验。

#### 实用指南
摘要提供项目页 https://dawn-parkour.github.io/ 并称含代码和视频；论文未明确模型权重或数据是否公开。复现需构造干净/带噪配对并实现两项训练损失。损失权重、温度、优化器和学习率未说明；噪声模型针对 RealSense D435i，迁移其他传感器需重新验证噪声和 sim-to-real 效果。

#### 总结
核心思想：噪声输入对齐干净几何
1. 为干净深度生成带噪配对。
2. 用带噪观测编码，以干净目标训练重建。
3. 对齐同场景两种潜在表示。
4. 用鲁棒状态驱动原策略控制。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29092v1)
- [arXiv](https://arxiv.org/abs/2609.29092v1)

---

<a id='2609.29644v1'></a>
## [Markerless Multi-Modal Autonomous Robotic Inspection of Large Space Structures](https://arxiv.org/abs/2609.29644v1)

**Authors:** Juan De Dios Alfaro, Arturo Ríos, David Rodríguez-Martínez, Carlos Pérez-del-Pulgar

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

Future orbital infrastructures, such as deployable antennas, solar farms, and large orbital platforms will require autonomous inspection systems able to operate with limited prior knowledge and without cooperative markers. Current on-orbit servicing approaches often rely on predefined trajectories, standard interfaces, fiducial markers or accurate target models, which limits scalability for large, heterogeneous or partially unknown structures. This paper presents a markerless autonomous robotic inspection pipeline in which 3D reconstruction is used as an inspection-support representation. The system integrates a Kinova Gen2 manipulator with an end-effector-mounted multimodal sensor head composed of an RGB-D camera, a thermal camera and a 2D LiDAR. The pipeline estimates an approximate inspection volume, generates viewpoints, plans collision-free motions with MoveIt, and synchronously records RGB-D images, thermal data, and robot poses in ROS2. Candidate reconstruction methods were evaluated to select a practical method for this pipeline, with Nerfacto used for geometric reconstruction and Thermal-Nerfacto used to demonstrate thermal-aware rendering for inspection. Validation in a Gazebo-based simulator and preliminary laboratory tests reveal that the proposed system can autonomously acquire spatially coherent inspection data and produce reconstructions suitable for visual and geometric assessment, representing a step towards inspection of large non-cooperative space structures.

### 论文解读

#### 摘要翻译
面向可展开天线、太阳能阵列和大型轨道平台等未来基础设施，论文提出无标记自主机器人巡检流程：用多模态传感器估计巡检范围、生成观察视点、规划无碰撞运动，并同步记录影像与机器人位姿。系统以 Nerfacto 重建几何，以 Thermal-Nerfacto 展示热感知渲染。Gazebo 仿真和初步实验室测试表明，它能采集空间连贯的数据并生成可供视觉、几何评估的重建结果。

#### 方法动机分析
大型、异构或部分未知结构难以预先建模；依赖合作标记、标准接口或固定轨迹是现有方案的痛点，也限制了扩展。作者的动机是先取得“够规划用”的粗几何，再让机械臂主动补采多视角数据；核心假设是粗体积与机器人位姿足以驱动采集。论文解决感知到可用数据的衔接，不是自动判定损伤。

#### 方法设计详解
整体流程的输入是传感器观测与机械臂位姿，输出是多视角数据和重建结果。原型使用 7 自由度 Kinova Gen 2，末端固定 RealSense D435i RGB-D、Optris PI 热相机和 RPLIDAR S3 二维 LiDAR。相机内参由平面标定板估计；手眼外参通过 AX=XB 求解，并用 ArUco 与 easy_handeye 标定。目标巡检时不需要标志物，但标定阶段仍需标志。机械臂沿已知轨迹移动 LiDAR，将扫描与机器人位姿组合成空间点云；过滤地面、天花板及无关点后，以 Open3D 提取目标近似包围盒。系统在可达区域围绕包围盒生成朝向中心的视点，由 MoveIt 求逆运动学并检查碰撞、关节限制。机器人停稳后同步采集 RGB、深度、热图和相机位姿，位姿含时间戳、平移及四元数，可整理为 Nerfstudio 数据。Nerfacto 用于主要几何重建，Thermal-Nerfacto 生成热感知渲染；论文没有展示 RGB-D、LiDAR 与热像完整融合的几何，也没有从热图自动诊断故障。推理/采集设置为机器人在每个视点停稳后再同步记录多模态数据；模型训练超参数未说明。

#### 方法对比分析
相较于依赖先验模型或预设轨迹的巡检，本文用现场 LiDAR 粗体积来驱动视点，再以统一的机器人规划和数据记录串起采集、重建。作者定性比较 COLMAP、CasMVSNet、MASt3R、NeuS、3D Gaussian Splatting 与 Nerfacto，选择 Nerfacto 是因其视觉效果、几何连贯性和集成实用性的折中；论文未给出比较评分或量化基准。因此主要贡献是系统集成，而非新重建算法。方案适合机械臂可达、传感器外参稳定且能获取多视角图像的场景。

#### 实验分析（精简版）
Gazebo 仿真使用全向移动底座实现目标周边多视角采集，并完成 Nerfacto 重建；实体实验使用固定底座，只覆盖大致正面扇区。两者均展示了采集和重建流程，另有热感知渲染示例。结果是定性的：论文未报告定量结果，无重建误差、覆盖率、规划成功率、延迟或温度精度等数字，也无消融指标，故只能说明原型链路可运行，不能据此量化精度或在轨可靠性。

#### 实用指南
复现需完成相机内参与手眼标定，建立 ROS 2 坐标变换，再依序实现 LiDAR 体积估计、可达视点生成、MoveIt 碰撞规划、停稳同步采集和 Nerfstudio 重建。文中提到 ROS 2、MoveIt、Open3D、Gazebo 等工具，但没有提供该项目代码、数据集或明确开源声明。视点间隔、滤波阈值、图像规格、训练超参数及算力未说明；迁移到其他机械臂或空间结构时需重做标定、工作空间配置，并验证光照、纹理和同步条件。仿真的移动底座不等同于自由漂浮航天器动力学。

#### 总结
核心思想：粗几何引导无标记多模态巡检
1. 用带位姿的 LiDAR 扫描圈定目标包围体。
2. 在可达范围布置朝向目标的视点。
3. 规划机械臂停稳采集并同步保存影像、位姿。
4. 用 Nerfacto 重建几何，以热分支呈现热感知视图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29644v1)
- [arXiv](https://arxiv.org/abs/2609.29644v1)

---

