time: 20260925

# Arxiv Computer Vision Papers - 2026-09-25

## Executive Summary

## 执行摘要

本期 10 篇论文呈现出鲜明的机器人系统导向：研究从单点感知或规划模块，延伸到传感器、在线决策与实体执行之间的完整链路。一个共同主题是让机器人在信息不完整、环境受限或仿真与现实存在差异时保持可靠：跨模态融合覆盖航空可见光—红外检测、Doppler LiDAR—惯性初始化、远程摄像头导航，以及视觉—触觉—音频操作；控制与规划工作则关注受限空间中的臂手协调、多机器人装配受力、持续挖掘和稀疏落脚点上的人形机器人跑酷。

值得重点关注的工作各自针对真实部署中的不同瓶颈。[CAMP](https://arxiv.org/abs/2609.29021v1) 将手部可行构型与机械臂路径联合搜索，并通过轨迹压缩和分层优化控制高维规划成本，适合关注灵巧操作与狭窄空间规划的读者。[WRAP](https://arxiv.org/abs/2609.29407v1) 把装配顺序、机器人分工和静力支撑统一到无夹具装配规划中，突出几何规划与接触力学的结合。[Free-Init](https://arxiv.org/abs/2609.29375v1) 利用 Doppler LiDAR 的逐点速度信息辅助惯性初始化，探索弱化扫描去畸变、激励运动和地图对应依赖的定位路线。感知方面，[FoCal](https://arxiv.org/abs/2609.29125v1) 按频段区分跨模态交互，并根据光谱差异调节融合；[ReVNM](https://arxiv.org/abs/2609.28976v1) 则从远程摄像头生成机器人自中心深度表征，展示了将环境基础设施纳入导航感知的可能性。

研究方向上，传感信息正从“增加模态”转向“按可靠性与任务阶段使用模态”：频域线索、时间记忆、接触信号及多普勒速度都被赋予明确的结构角色。另一个趋势是轻量化适配——在线故障检测以边缘模型递推更新，扭矩观测对齐以低成本校准支撑零样本抓取迁移。与此同时，挖掘、装配、导航和跑酷等工作更多在物理平台上检验系统闭环，而不止报告离线指标。

若优先阅读全文，建议从 CAMP（受限操作规划）、WRAP（受力约束的多机器人装配）、Free-Init（新型速度传感与状态初始化）和 Echo in the Steps（深度感知、门控记忆与人形步态）入手；研究多模态感知的读者可进一步关注 FoCal 与 PolyUMI。

---

## Table of Contents

1. [FoCal: Frequency-Oriented Cross-Modal Interaction and Spectral Calibration for Aerial Visible-Infrared Object Detection](#2609.29125v1)
2. [CAMP: Cooperative Arm-Hand Motion Planning in Constrained Spaces](#2609.29021v1)
3. [WRAP: Fixtureless Wrench-aware Multi-Robot Assembly Planning](#2609.29407v1)
4. [Continuous Online Fault Detection for Mobile Robots via Adaptive Edge Models](#2609.29194v1)
5. [PolyUMI: Accessible Visual-Tactile-Audio Data Collection for Object Inference and Manipulation](#2609.29760v1)
6. [From Target Selection to Digging: A Learning-Based Framework for Continuous Autonomous Excavation](#2609.29750v1)
7. [Free-Init: Scan-Free, Motion-Free, and Correspondence-Free Initialization for Doppler LiDAR-Inertial Systems](#2609.29375v1)
8. [Simple Torque-Observation Alignment for Zero-Shot Sim-to-Real Grasping with a Direct-Drive Gripper](#2609.29031v1)
9. [ReVNM: Learning-Based Visual Navigation from a Remote Camera](#2609.28976v1)
10. [Echo in the Steps: Learning Perceptive Humanoid Parkour with Gated Memory](#2609.28960v1)

---

## Papers

<a id='2609.29125v1'></a>
## [FoCal: Frequency-Oriented Cross-Modal Interaction and Spectral Calibration for Aerial Visible-Infrared Object Detection](https://arxiv.org/abs/2609.29125v1)

**Authors:** Ben Liang, Chao Sui, Junqi Bai, Yuan Liu, Chunlai Li, Xiubao Sui, Qian Chen

**Published:** 2026-09-24

**Categories:** cs.CV

**Abstract:**

In aerial RGB--IR object detection, effectively exploiting complementary information across modalities is critical for robust perception under complex illumination and environmental conditions. Existing multimodal detectors mainly focus on spatial-domain interaction or frequency-specific feature enhancement, while the cross-modal interaction patterns of different frequency components remain insufficiently explored. Moreover, spectral discrepancy itself may contain both useful complementary cues and unreliable modality-specific responses, making indiscriminate frequency fusion suboptimal. To address these issues, we propose FoCal, a frequency-oriented framework for aerial RGB--IR object detection. First, a Frequency-Aware Dual-Domain Calibration (FADC) module is developed to explicitly model frequency-dependent cross-modal interaction. Low-frequency components are collaboratively consolidated into a shared structural consensus, whereas high-frequency components preserve modality-specific information through selective cross-modal exchange. The resulting frequency-aware cues are further transferred to the original feature domain to regulate cross-modal calibration. Second, we introduce a Discrepancy-Guided Spectral Modulation (DGSM) module, which characterizes cross-modal spectral imbalance using confidence-weighted relative amplitude discrepancy and transforms it into a bounded signed gate for adaptive enhancement, preservation, or attenuation of the joint multimodal spectrum. Extensive experiments on DroneVehicle, ESCVehicle, and ATR-UMOD demonstrate the effectiveness of FoCal, yielding $\mathrm{mAP}_{50}$ values of 83.5\%, 54.8\%, and 64.6\%, respectively. Meanwhile, with only 3.0M parameters, FoCal achieves 113.6 FPS while preserving leading detection accuracy, highlighting a favorable accuracy--efficiency trade-off. Code is available at {https://github.com/universeliang/FoCal.

### 论文解读

#### 摘要翻译
航空可见光—红外检测要在复杂光照和天气下利用两种传感器的互补信息。FoCal 指出现有方法常在空间域融合，或统一增强频率，却没有区分不同频带的交互方式；同时模态间频谱差异可能是有效线索，也可能是噪声。为此，FoCal 以频率感知双域校准（FADC）建模频带交互，以差异引导光谱调制（DGSM）按置信度调节联合频谱。在 DroneVehicle、ESCVehicle、ATR-UMOD 上分别取得 83.5%、54.8%、64.6% 的 mAP50，模型为 3.0M 参数、113.6 FPS。

#### 方法动机分析
低频通常承载轮廓和整体结构，高频对应边缘与局部细节。若所有频率都用同一种融合规则，容易让有用的模态特性被冲淡；若直接融合两路频谱，也难区分互补差异与不可靠噪声。论文的关键假设是：低频宜建立共享结构，高频宜保留各模态特点并选择性交换，频谱差异则应结合能量置信度决定增强或抑制。方法针对配对、对齐的航空 RGB-IR 目标检测。

#### 方法设计详解
输入为可见光与红外图像对，双流骨干提取多尺度特征，在 P5 共享高层语义以削减冗余；FADC、DGSM 用于 P3、P4。FADC 用 Haar 小波把特征拆为低频近似项和三个方向的高频项。低频按学习权重加权成共同结构锚点；高频则双向传递，接收掩码控制对方细节注入，避免简单相加。重组后的频率特征再生成空间、通道可靠性掩码，校准回原特征流。DGSM 将两路特征投影后作 FFT，计算归一化幅度差，并以谱能量置信项加权；可学习映射和 tanh 生成有符号门控，正值增强、负值衰减联合频谱，再经逆变换回到空间域并残差融合。最终由 PAN 汇聚尺度信息，检测头输出目标框和类别。

#### 方法对比分析
FADC 的主要差异在于频率相关的交互拓扑：“低频共识、高频选择交换”，而非全频统一融合。DGSM 也不同于固定频谱增益，它根据相对差异及置信度进行双向调制。另有 RR 共享高层语义以降低参数冗余，属于轻量化结构策略。该设计适用于传感器配准良好的成对检测输入；非对齐或视频场景的有效性尚未验证。

#### 实验分析（精简版）
论文在三个航空车辆数据集上报告 COCO 风格 mAP50/mAP。DroneVehicle 上 FoCal 为 83.5%，高于 C²DFF-Net 的 82.0%；ESCVehicle 为 54.8%，高于 C²-VeD 的 52.4%；ATR-UMOD 为 64.6%，略高于 COMO 的 64.2%。DroneVehicle 消融中，双流基线为 80.8%，加 RR、FADC、DGSM 后完整模型达到 83.5%。复杂度为 3.0M 参数、8.4G FLOPs；RTX 4090、batch size 1 下延迟 8.8 ms/对。结果显示精度与速度兼顾，但论文未专门评估 TensorRT 或混合精度部署，也未验证时序场景。

#### 实用指南
实现基于 Ultralytics 扩展，模型从头训练：150 epoch、batch size 16、SGD 初始学习率 0.01，动量 0.937、权重衰减 0.0005。复现需准备配对图像及检测标注，并保证可见光与红外输入对齐；部署时应在目标硬件上重新测时。作者提供代码：https://github.com/universeliang/FoCal。迁移到其他传感器或类别时，需适配标注和类别并重新训练、验证频率交互是否仍然合适。

#### 总结
核心思想：按频带分工并校准融合
1. 双流提取两种模态特征，共享高层语义。
2. 小波分解，低频形成共识、高频选择性交换。
3. 用可靠性掩码校准跨模态回注。
4. 根据置信度加权频谱差异，正负门控调制联合频谱。
5. 多尺度汇聚后预测框与类别。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29125v1)
- [arXiv](https://arxiv.org/abs/2609.29125v1)

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
协调机械臂与灵巧手运动，是复杂受限环境中灵巧操作的基础。分开规划臂与手容易漏掉必须协同调整的解；直接在高维联合空间搜索又很困难。论文定义每个臂姿态对应的可行手型集合，并据此提出 CAMP：先分层搜索手部路径，受阻时局部调整臂，再以端点保持的运动基元表示轨迹并粗到细联合优化。六项仿真成功率为 84.2%–98.5%，并验证实体可执行性。

#### 方法动机分析
臂和手的无碰撞构型并不能独立处理：臂姿态变化会改变可用手型，固定手型会排除一些可行臂路；反过来，逐层存在可行手型也不保证相邻层之间能连续连接。论文的核心假设是，先用低维臂路径提供多样全局引导，再根据路径位置搜索变化的可行手纤维，遇到连接障碍时小幅放松臂姿态，可避免盲目联合搜索的维度负担并修复僵化分解的失败。

#### 方法设计详解
输入是机器人臂手模型、已知障碍及起终点构型。RRT-Connect 先在臂空间生成路径并分层；双向 Hand-RRT 沿各层对应的无碰撞手型集合寻找从初始手型到目标手型的连接。若相邻层插值中点碰撞，局部重规划同时调整手构型和臂关节偏移，在邻近臂姿态诱导的新可行集合中寻找过渡。

初始化轨迹再编码为 VMP：每个时刻的关节值由端点线性参考、带高斯基函数的形变和端点包络组成；包络在起终点为零，因此优化权重不会破坏端点。系统含 6 自由度 UR7e 和 16 自由度 LinkerHand，轨迹用 200 个点；臂、手每关节分别用 30、20 个基函数，总计 500 个权重，远少于逐点表示的 4,356 个变量。目标兼顾轨迹平滑质量、相对初始路线的偏离和基于有符号距离场的碰撞惩罚，并受关节位置、速度及碰撞约束。规划先运行 16 个初始化实例、每路最多 10 秒和 20 层；臂偏移限于 ±0.1 rad，最多保留 8 个初始候选。粗阶段并行优化至多 100 次迭代，选出前 3 个候选；细阶段再优化最多 200 次迭代，重点提升轨迹质量和可行性。

#### 方法对比分析
Arm-Then-Hand 固定手型规划臂，预设手型法仍难处理途中变手型。RRT-Connect 搜索完整联合空间，QRRT* 分层简化机器人，CHOMP 与 A*+CHOMP 依赖单一初始化。CAMP 将多个臂空间引导、纤维上的分层手搜索、局部臂松弛和紧凑 VMP 联合优化组合起来，兼顾路线多样性和协同细化，适用于已知静态障碍下的狭窄臂手操作。

#### 实验分析（精简版）
六项仿真任务与 RRT-Connect、QRRT*、CHOMP、A*+CHOMP 对比，每个任务—方法进行 10 批、每批 100 次试验，并用 MuJoCo 做最终碰撞验证。CAMP 在窄通道成功率为 92.50%，对比 A*+CHOMP 的 71.50% 和 RRT-Connect 的 3.00%；柜体圆柱预抓取成功率为 84.20%，高于 A*+CHOMP 的 18.80%，平均规划 42.60 秒（对方 98.40 秒）。Ball-in-Box 消融中，去掉臂松弛后成功率由 92% 降为 74%；逐点表示耗时 84.97 秒、成功率 79%，完整 VMP 法耗时 32.45 秒、成功率 92%。真实机器人各做 10 次，球盒、按键目标到达、柜体圆柱成功率为 80%、90%、80%；按键实验未实际按下按钮。结果支持可行性提升，但平台只有一套，且真实执行仍受定位、标定和执行误差影响。

#### 实用指南
论文给出项目网站，但未明确说明代码、模型及数据的开放许可。该方法不训练神经网络；复现设置为 200 个轨迹点、臂/手 30/20 个基函数、16 路初始化、20 层、±0.1 rad 臂松弛、粗/细 100/200 次迭代。原评估用 PyRoKi 的 SDF 与胶囊近似规划，再以 MuJoCo 精确验碰；迁移到新机械臂或手时需替换运动学、碰撞模型、目标构型生成和验证器，并重新调节层数、偏移范围及基函数设置；动态障碍和不确定性尚未处理。

#### 总结
核心思想：沿可行手纤维协同规划
1. 在臂空间产生多条全局引导。
2. 沿途分层搜索可行手型并连接。
3. 碰撞时局部调整臂手，修复过渡。
4. 用端点保持 VMP 压缩轨迹、粗筛候选。
5. 细化优选轨迹并进行精确碰撞验证。

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
机器人装配常依赖定制夹具或仅自上而下的操作。多机器人协作有望减少夹具并提高灵活性，但装配顺序和任务分配组合很多，还必须判断装配反作用力是否需要额外机械臂或桌面支撑。WRAP 接收部件依赖关系、网格和初始状态，以线性规划检验抓取支撑能力，再用便宜的反向搜索启发式引导逐步的正向搜索；随后求解多机器人运动规划，并将接触装配与自由运动分开执行。

#### 方法动机分析
传统方案常固定机器人角色、装配方向或定制夹具，难适应高混合、低重复任务；忽略力的几何规划也可能让压配部件滑动或倾倒。WRAP 假设装配依赖及所需作用力已知，且抓取器、部件接口和桌面可承受的力能用集合表示。若各步骤存在满足静力平衡的支撑组合，就可让协作机器人或桌面充当临时夹具。它关注任务和支撑可行性，不从零推断装配关系。

#### 方法设计详解
系统输入部件网格、初始位姿、装配依赖图和装配作用力。它先为各子装配求稳定放置姿态，再采样二指对向抓取或吸盘法向抓取。候选装配步骤建立每个部件的静力平衡：机器人和桌面的支撑 wrench、连接接口可传递的 wrench，与重力及装配外力相加为零。线性规划在各能力集合内寻找可行分配；为容忍误差，还把横向力扰动和作用点偏移纳入外力集合，并要求每个顶点都能平衡。搜索状态涵盖已完成装配、桌面放置和机器人抓取，动作包括抓取、放置、装配、交接和释放。依赖图将搜索切成装配里程碑；反向拆解搜索用碰撞和力约束检查，并把抓取归并成“能否传递所需装配力”的类别，用 Dijkstra 得到启发式，再引导正向贪心搜索。找到初解后，搜索逐步扩大装配窗口并以精确 A* 降低动作成本；关键帧经动态规划减少关节运动并增大机械臂间隙。仿真装配力为 10 N；真实规划名义力为 25 N。

#### 方法对比分析
不同于仅凭几何排程或忽略力的协作规划，WRAP 把承力检查纳入机器人分工、抓取、交接、放置和装配步骤。相较 Fabrica 固定部分装配姿态的做法，WRAP 可重抓、重定向、交接或暂放桌面，并显式验证装配力是否有支撑；代价是依赖给定的装配图与已建模的力能力。它适用于需要多臂协作或临时支撑的装配，不适用于非抓取推移等尚未建模的操作。

#### 实验分析（精简版）
实验覆盖自建椅凳、十字、立方体和多种 Fabrica 装配；仿真大多使用四台机械臂，每种设置重复十次。Cross 上，WRAP 初始方案平均耗时 2.01±0.02 秒，Optimistic 基线为 11.25±0.15 秒，二者动作成本均为 22。Stool 上 WRAP 用时 1.60±0.02 秒、成本 32，基线用时 14.42±0.09 秒、成本 20，说明快速找到方案不等于动作最少。MuJoCo 支撑消融中，忽略装配力或只考虑重力的 Cross、Cube、Chair 成功率均为 0%；完整方法分别达到 100%、90%、100%。真实系统由两台 UR5e 完成七动作半凳装配，包含一次交接和两次插入。优势是力约束避免缺少支撑的方案；局限是搜索随机器人数量增长，真机尚未闭环重规划。

#### 实用指南
代码、视频和三维模型见论文项目页 vhartmann.com/wrap；论文未说明独立数据集。复现需准备部件网格、初始位姿、依赖图、装配力，并针对所用夹爪实测承载边界。真实规划采用 25 N 名义力，来自最大实测轴向 17.32 N、横向 5.38 N 并留有余量；仿真使用 10 N。迁移到新任务需重新建立几何、依赖、机器人与接口 wrench 能力、稳定放置姿态及接触控制器；不同材料和夹具应重新标定。

#### 总结
核心思想：以力平衡安排协作支撑。
1. 从依赖图建立装配里程碑和可用抓取。
2. 用静力平衡筛选机器人、接口及桌面支撑。
3. 反向承力类别搜索启发正向装配动作。
4. 扩大搜索窗口优化动作，并生成多臂轨迹。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29407v1)
- [arXiv](https://arxiv.org/abs/2609.29407v1)

---

<a id='2609.29194v1'></a>
## [Continuous Online Fault Detection for Mobile Robots via Adaptive Edge Models](https://arxiv.org/abs/2609.29194v1)

**Authors:** Jordan Levy, Nicolas Verstaevel, Vincent Talon, Benoit Gaudou

**Published:** 2026-09-24

**Categories:** cs.RO, cs.LG

**Abstract:**

Mobile robots require robust, real-time fault detection capable of continuous adaptation on constrained edge hardware. While deep time-series models excel at unsupervised anomaly detection, their computational cost prohibits high-frequency onboard execution. This paper bridges this gap via a Teacher-Student distillation framework. An offline foundation model (TSPulse) generates pseudo-labels from unlabeled time series augmented with fault injections. A lightweight MiniRocket Student, adapted with a Recursive Least Squares estimator, approximates this complex decision boundary to execute real-time inference onboard. Evaluations on the TSB-AD benchmark and a physical mobile robot demonstrate the Student achieves a 4.30 ms CPU inference latency. During real-world domain shifts, online adaptation enables the Student to recover from unseen mechanical degradation, improving VUS-PR scores from 0.26 to 0.75 without catastrophic forgetting. Crucially, an uncertainty-guided active learning strategy minimizes operator cognitive load, requesting sparse interventions only when encountering novel fault distributions. These results validate the deployment of state-of-the-art anomaly detection on resource-constrained robotics through offline-to-online distillation.

### 论文解读

#### 摘要翻译
移动机器人需要能在边缘硬件上持续适应的实时故障检测。深度时间序列模型虽擅长无监督异常检测，却难以高频机载运行。本文以 TSPulse 离线处理无标签序列并生成伪标签，再蒸馏给轻量 MiniRocket 学生；学生用递归最小二乘（RLS）在线适配。基准与实体机器人实验中，学生 CPU 推理为 4.30 ms；真实域偏移下 VUS-PR 从 0.26 提升到 0.75，且未出现灾难性遗忘。不确定性门控仅在新颖故障时请求人工反馈。

#### 方法动机分析
固定检测器会受磨损、载荷与环境变化影响；大模型推理昂贵，持续反向传播还会增加边缘计算负担并可能遗忘旧模式。作者假设教师产生的连续异常分数足以监督紧凑学生，而固定随机特征加在线线性头可在保持快速推理的同时适应新故障。目标是检测与适应，不是故障根因诊断。

#### 方法设计详解
流程的输入是历史传感器窗口，输出为连续异常分数及可在线更新的学生检测结果。教师 TSPulse 对窗口评分，分数经滚动平均平滑、z-score 标准化，并以指数权重强调严重异常。为扩展训练覆盖，作者向序列注入尖峰、截断、漂移和噪声，再训练 MiniRocket 回归教师分数。MiniRocket 以固定卷积核把窗口映射为高维特征，统计响应为正的比例；学生仅需学习特征到异常分数的映射。在线推理阶段用 RLS 替换静态 ridge 头：利用预测误差和协方差逆矩阵更新权重，无需反向传播。遗忘因子为 0.99；不确定性由特征空间新颖度与预测方差共同决定，只有乘积达到阈值才触发反馈和更新。实验采用 TSPulse 窗口 512、学生窗口 100、2,000 个核；机器人数据含 63 个通道、50 Hz 采样。

#### 方法对比分析
直接运行 TSPulse 可保留复杂检测能力但延迟较高；蒸馏后的 MiniRocket 更轻。与静态学生相比，RLS 可针对新分布修改线性头；固定特征也避免全模型在线训练。新意在于把教师伪标签、故障注入、递推适配与不确定性门控串成边缘检测流程，组件本身并非全新。它适合有历史无标签数据、持续传感器流和稀疏人工确认的场景；成效取决于教师标签及注入样本能否覆盖目标故障。

#### 实验分析（精简版）
评估涵盖 TSB-AD 与校园四轮机器人数据，指标为兼顾迟报和持续误报的 VUS-PR；真实故障包括过量耗电、超载与 GNSS 干扰。Jetson Orin Nano Super CPU 上，MiniRocket（窗口 100）平均延迟 4.30 ms，相比 TSPulse（窗口 512）的 343.45 ms 大幅降低。域偏移实验中，在线适配把 VUS-PR 从静态/离线基线的 0.26 提至 0.75；2,000 核的 RLS 更新为 12.08 ms。合成注入优于所比较的 VAE 和混合增强，增强量为 6 左右后收益趋平。优势是较快且可适应；局限是缺乏多小时连续转换评估、特征不易解释，统计型合成扰动也未必代表复杂物理故障。

#### 实用指南
论文提供代码和复现数据：https://anonymous.4open.science/r/ICRA2027-AB08。复现时先生成并平滑教师分数，再做加权蒸馏与四类故障注入；部署时冻结 MiniRocket 特征，以 RLS 更新，并依据不确定性阈值控制人工反馈。文中给出遗忘因子 0.99、2,000 个核及 Jetson CPU 延迟。迁移至新机器人需调整传感器窗口、标准化和故障注入，并重新训练学生、校准阈值。论文未说明全部依赖版本和通用阈值选择流程；较大特征维度会增加 RLS 更新耗时。

#### 总结
核心思想：蒸馏后按不确定性在线适配。
1. TSPulse 给无标签序列打分并平滑。
2. 注入故障模式，训练 MiniRocket 拟合教师。
3. 固定特征，用 RLS 递推更新学生线性头。
4. 不确定性越阈时才请求确认并学习新分布。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29194v1)
- [arXiv](https://arxiv.org/abs/2609.29194v1)

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
人类利用视觉、触觉、听觉和本体感觉调整操作；多数模仿学习只记录视觉与位姿，难获被遮挡的接触信息。本文提出开源平台 PolyUMI：无线手持夹爪同步记录腕部视觉、光学触觉、接触音频和本体状态；同一传感手指可转装到机器人末端，保持示教和执行的感知几何一致。配套的 VisTA 将多模态、多时刻观测融合为接触感知动作。实验表明触觉和音频能补充视觉，VisTA 在接触丰富任务中有竞争力。

#### 方法动机分析
相机难直接观察指尖局部形变，也可能错过滑移、碰撞或旋紧到位等短暂事件。已有接口通常只整合触觉或音频；PolyUMI 将视触听集成进无线示教设备，并复用传感手指以减小域偏移。作者的关键假设是，视觉提供全局场景，触觉与结构振动揭示接触状态，保留这些信号的空间和时间细节后联合推理，能改善需要持续接触调节的任务。

#### 方法设计详解
手持夹爪内置电池、计算机和音频接口；可移装的传感指以内部相机观察七层 VHB 胶带和铝粉反射层形成的触觉表面（20 fps），压电麦克风记录 16 kHz 单声道接触音频；GoPro Hero 12 以 60 fps、1920×1080 记录约 177° 腕部视野。手持位姿由相机和 IMU 上的 ORB-SLAM3 估计，机器人端使用关节状态，两端都记录夹爪宽度。不同频率的信号按统一时间轴对齐，并以 10 Hz 插值。VisTA 使用当前及前一时刻（H=2）：图像缩放至 224×224，约 0.5 秒音频转为 128 频带 log-Mel 谱（3×128×48），机器人状态为 16 维。独立 CNN 将视觉、触觉编码为带空间信息的 token，音频 CNN 输出 96 个 token，状态由 MLP 编码；嵌入维度 624、合计 294 个 token。八层、八头 Transformer 对 token 做双向自注意力，使局部触觉、短音频事件和视觉上下文跨模态交互。融合表征经 cross-attention 条件化 18 层 DiT 风格动作头，以条件流匹配预测 16 步动作块；每步 10 维，包含相对平移、旋转和夹爪宽度。控制执行前三步后重新观测；延迟通过校准决定跳过动作数，机器人再以 1 kHz 阻抗控制执行插值轨迹。

#### 方法对比分析
相较视觉为主的 UMI，PolyUMI 同步加入光学触觉和接触音频；相较 PolyTouch，重点是独立开源、无线手持采集和传感手指跨手持夹爪/机器人复用。VisTA 保留带空间、时间标记的细粒度 token，再以全序列自注意力融合，不像将每种模态先压成单个向量；之后由 flow-matching 动作头生成动作。基线包括视觉 Diffusion Policy、MulSA、Sparsh-X 与 PolyTouch，覆盖视觉策略及不同多模态融合方式。接触遮挡、滑移和持续贴合更可能受益；若任务状态本来清晰可见，多传感器未必增加效果。

#### 实验分析（精简版）
五类触觉形状的 2700 张图像按场景划分，留出 714 张测试；分类器训练 120 epoch、RTX 4060 上约 11 分钟，准确率 92.3%。闭盒识别三类物体时，每类用 20 次示范训练、10 次验证；不含音频的视觉/触觉组合约 42–44%，接近随机猜测 33%，加入音频后约 80%。防滑控制中视触听策略成功 8/10 次，视觉策略 2/10，触觉加音频 7/10。约 25 cm 擦板任务中 VisTA 成功率约 90%，Sparsh-X 约 40%，视觉 Diffusion Policy 与 MulSA 约 20–25%；前者更能保持接触并沿线擦除。灯泡旋入时所有方法至少 80%，视觉基线最高、VisTA 接近，说明接触模态的收益依任务而异。真机验证覆盖的任务数量有限，跨机器人泛化尚未展示。

#### 实用指南
项目页公开硬件设计、电子、固件、制造说明和学习软件；论文没有明确说明数据集是否公开。多模态指约增加 240 美元、装配约 4 小时，传感指换装约 10 分钟。复现需实现统一时间戳、10 Hz 同步和手持位姿估计，并按论文构建两帧图像、0.5 秒音频窗及 token 策略。触觉分类训练 120 epoch；优化器、学习率和 batch size 未说明。迁移时须校准传感延迟、末端坐标及夹爪控制，并为新任务另采示范、重新训练。

#### 总结
核心思想：**同一传感指连通多模态示教与执行**。
1. 手持端同步采集视、触、听和位姿。
2. 把异步信号对齐并编码为细粒度 token。
3. 用 Transformer 融合模态及时间线索。
4. 流匹配动作块，校准延迟后闭环执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29760v1)
- [arXiv](https://arxiv.org/abs/2609.29760v1)

---

<a id='2609.29750v1'></a>
## [From Target Selection to Digging: A Learning-Based Framework for Continuous Autonomous Excavation](https://arxiv.org/abs/2609.29750v1)

**Authors:** Shuai Zhao, Ji-an Pan, Quantao Yang, Zheng Wang, Chaoyi Chen, Qing Xu, Keqiang Li

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

Repeated excavation continuously reshapes pile geometry, requiring an autonomous excavator to adapt its digging targets and coordinate motion across successive excavation cycles. We present a learning-based framework for continuous autonomous excavation that integrates terrain-aware target selection with reinforcement- and imitation-learning controllers. The framework separates target-conditioned motion from local digging: a shared task-conditioned RL policy controls waypoint-guided approach and loaded transport, while an IL policy learns vision-based digging and lifting from expert demonstrations. Digging targets are selected from LiDAR elevation maps and converted into bucket-tip waypoints for motion control. The control architecture coordinates the learned policies and deterministic unloading through a shared motion interface. The complete system is deployed on a scaled hydraulic excavator with multimodal sensing and closed-loop actuator control. Offline replay and physical experiments demonstrate more consistent target selection, shorter local motion time, and increased payload compared with the respective baselines. The learned digging policy achieves a mean payload of 6.52 kg per completed cycle, compared with 2.68 kg for Fixed Dig. Three five-scoop runs further demonstrate consecutive autonomous excavation under continuously changing pile geometry.

### 论文解读

#### 摘要翻译
重复挖掘改变料堆形状，要求系统连续更新目标并协调运动。本文提出学习式框架：地形感知模块从激光雷达高程图选择目标，共享强化学习策略负责路径点引导的接近和载料运输，模仿学习策略则从专家示范学习视觉挖掘与提升。斗尖路径点连接目标与运动，共享接口协调策略及固定卸料；系统部署于具多模态传感和闭环控制的缩比液压挖掘机。实验报告了更稳定的选点、更快的局部运动，以及相较固定挖掘更高的单循环载荷；三次五铲连续运行验证了料堆变化下的无人作业。

#### 方法动机分析
每次取土都会改变下一次作业面对的地形，逐帧选择最高点容易受观测噪声影响；而接近目标和实际挖土又有不同控制目标。前者适合路径跟踪，后者需要协调斗杆、铲斗的插入、卷挖和提升。液压死区也会让小幅指令无法产生预期动作。核心假设是先把铲斗带到合适预挖状态，局部示范技能即可复用于变化目标，无需输入绝对地形目标。

#### 方法设计详解
流程从激光雷达10厘米高程网格开始，以表面与地面高度差估计土层厚度，过滤低置信度、过陡或不合适的区域，再按厚度、置信度、边界间隙、相对高度和粗糙度评分。跨帧关联、连续三帧确认、0.10分数迟滞和最多十帧缺测保留，抑制目标跳变。目标转成斗尖路径点后，共享PPO控制策略按任务模式执行空斗接近或载料运输；其38维观测包括液压缸、斗尖、路径误差、任务模式和斗角状态，并惩罚偏离路径、动作突变及死区内无效指令。接近和运输的目标斗角分别为70度与180度。PPO策略采用两层、每层256单元的MLP，训练步数为200万。进入预挖位后，ACT读取640×480 RGB图像和11维本体状态，输出20步动作块，每执行10步便用新观测更新。结束后由RL运输、固定策略卸料，再观测地形进入下一循环。

#### 方法对比分析
与全局最高点或仅按空间特征评分相比，时间目标选择器显式维护历史并使用迟滞，以时间一致性换取少量无输出帧。与DLS运动控制相比，PPO学习液压响应，并共享用于两种目标条件运动。与固定缸位置挖掘相比，ACT从专家数据学习局部视觉动作。主要贡献是把可更新的几何目标、路径跟踪和可复用的局部挖掘技能组合成闭环作业流程；这种设计仍要求预挖姿态落在示范技能适用范围内。

#### 实验分析（精简版）
离线回放中，时间目标选择器在1041对相邻有效输出中出现1次XY跳变，最高点法为300次，空间法为523次；目标输出率为98.68%。五次10厘米局部运动均达到2厘米位置容差，PPO平均耗时1.04±0.54秒，DLS为1.93±0.11秒。以相同目标序列完成的20个循环里，ACT平均载荷6.52千克，固定挖掘为2.68千克。实机完成三次五铲运行，共转运94.75千克，未需人工干预。证据覆盖单一土类、静止的缩比机器；选点对照主要证明回放稳定性，尚不足以证明跨工地泛化。

#### 实用指南
论文未明确提供代码、模型或数据的公开仓库。复现需实现10厘米地形网格、候选筛选与跨帧迟滞，按论文设定训练PPO，并以83段训练、21段验证的遥操作序列训练ACT；部署时20步预测、执行10步后更新观测。还须标定坐标系、斗尖运动学、液压方向相关速度限制和死区，并在策略切换时确认零指令。迁移到不同土质、机器尺度或移动底盘时，应重新标定并采集覆盖目标工况的示范，再验证预挖姿态和载荷表现。

#### 总结
核心思想：以几何目标连接运动与挖掘
1. 从地形评分选点，并跨帧确认稳定目标。
2. 将目标转为斗尖路径点，由PPO控制接近。
3. ACT依据图像与机身状态分段完成挖掘、提升。
4. RL运输、固定卸料，再观测地形开始下一铲。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29750v1)
- [arXiv](https://arxiv.org/abs/2609.29750v1)

---

<a id='2609.29375v1'></a>
## [Free-Init: Scan-Free, Motion-Free, and Correspondence-Free Initialization for Doppler LiDAR-Inertial Systems](https://arxiv.org/abs/2609.29375v1)

**Authors:** Mingle Zhao, Jiahao Wang, Tianxiao Gao, Chengzhong Xu, Hui Kong

**Published:** 2026-09-24

**Categories:** cs.RO, cs.CV, eess.SY

**Abstract:**

Robust initialization is crucial for online systems. In the letter, a high-frequency and resilient initialization framework is designed for LiDAR-inertial systems, leveraging both inertial sensors and Doppler LiDAR. The innovative FMCW Doppler LiDAR opens up a novel avenue for robotic sensing by capturing not only point range but also Doppler velocity via the intrinsic Doppler effect. By fusing point-wise Doppler velocity with inertial measurements under non-inertial kinematics, the proposed framework, Free-Init, eliminates reliance on motion undistortion of LiDAR scans, excitation motions, and map correspondences during the initialization phase. Free-Init is also plug-and-play compatible with typical LiDAR-inertial systems and is versatile to handle a wide range of initial motions when the system starts, including stationary, dynamic, and even violent motions. The embedded Doppler-inertial velocimeter ensures fast convergence and high-frequency performance, delivering outputs exceeding 10 kHz. Comprehensive experiments on diverse platforms and across myriad motion scenes validate the framework's effectiveness. The results demonstrate the superior performance of Free-Init, highlighting the necessity of fast, resilient, and dynamic initialization for online systems.

### 论文解读

#### 摘要翻译
Free-Init 面向 FMCW 多普勒激光雷达与 IMU 组成的里程计系统，提出高频、鲁棒的初始化框架。方法融合逐点径向多普勒速度和惯性测量，在初始化阶段免扫描去畸变、免特定激励动作、免地图对应，可用于静止、动态乃至剧烈运动启动，并可接入常见 LIO。

#### 方法动机分析
很多 LIO 初始化默认设备静止，或依赖先去畸变的扫描和地图匹配；车辆已经行驶、手持设备剧烈运动，或环境缺少几何特征时，这些条件难以满足。要求人为旋转、加速的方案也不适合所有平台。Free-Init 的核心假设是 FMCW 雷达能为静态回波提供逐点多普勒速度，且雷达与 IMU 刚性连接、外参已标定；以运动学直接从这些观测恢复启动状态。

#### 方法设计详解
输入包括逐点雷达多普勒读数、陀螺仪和加速度计。首先，12 维 Doppler-Inertial Velocimeter（DIV）状态包含雷达及机体线速度、机体角速度和陀螺偏置。每个静态点的径向速度约束雷达速度在视线方向上的投影，陀螺读数约束角速度与偏置之和；滤波器逐点更新，残差阈值用于剔除动态点。接着，方法把估计速度、加速度计读数和非惯性运动学关系放入优化，求加速度计偏置与重力；相对加速度约束同时考虑角速度、角加速度和科氏项，重力模长设为 9.81 m/s²。姿态在初始化窗口中按固定轴旋转假设积分，点云据此投影成初始地图，再将位姿、速度、偏置、重力和地图交给后续 LIO。推理时，DIV 逐点输出频率超过 10 kHz。

#### 方法对比分析
与先去畸变再做扫描匹配的初始化相比，关键差别是把雷达多普勒作为直接速度观测，不等待整帧几何对应；相较静止启动或人为激励法，它支持非零初速且不要求操作者执行特定动作。重力和加速度计偏置优化、姿态积分及建图属于后续估计环节。方法适用于具备逐点多普勒测量的 FMCW 雷达，不适用于普通 ToF 雷达；它仍依赖足够静态回波、有效外参及初始化中的固定轴旋转近似。

#### 实验分析（精简版）
实验使用 Aeva Aeries II 雷达和 Xsens MTi-G-710 IMU，覆盖手持、轮式与车辆平台的动态和静止序列，并把初始化接入 FAST-LIO2、DLIO，以绝对平移 RMSE、端到端误差和运行时间评估；速度对照 RTK/INS。车辆启动速度约 60–75 km/h。FAST-LIO2 在 dyna_01 的 RMSE，默认初始化为 8.21 m、Free-Init 为 0.25 m（RMSE 由 8.21 到 0.25 m）；在 dyna_02 默认法失败，而 Free-Init 为 0.10 m。DLIO 在 dyna_05 的 RMSE，默认初始化为 36.90 m、Free-Init 为 1.10 m（RMSE 由 36.90 到 1.10 m）。动态、静止序列平均运行时间分别为 0.208 s、0.116 s。结果显示高动态启动有明显收益；实验硬件和序列有限，跨设备泛化及极端动态点比例的系统评估仍不足。

#### 实用指南
论文资源栏给出 Free-Init、FMCW-LIO 代码与数据序列链接及实验视频。复现需要逐点多普勒 FMCW 雷达、同步 IMU 和离线标定的雷达—IMU 外参，依次实现逐点滤波、动态点筛除、重力/加速度计偏置优化和地图初始化。论文报告的硬件为 Aeva Aeries II 与 Xsens MTi-G-710；完整噪声参数和所有优化实现细节未说明，需从代码确认。迁移平台时应重新标定外参并检查时间同步、静态点假设与固定轴近似；换成 ToF 雷达则须替换核心多普勒观测机制。

#### 总结
核心思想：多普勒速度直接初始化
1. 逐点多普勒与陀螺观测更新 DIV。
2. 估计雷达/机体速度、角速度和陀螺偏置。
3. 以非惯性加速度约束求重力与加速度计偏置。
4. 积分姿态、生成初始地图并交给 LIO。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29375v1)
- [arXiv](https://arxiv.org/abs/2609.29375v1)

---

<a id='2609.29031v1'></a>
## [Simple Torque-Observation Alignment for Zero-Shot Sim-to-Real Grasping with a Direct-Drive Gripper](https://arxiv.org/abs/2609.29031v1)

**Authors:** Doyoung Kim, Edgar Lee, Hyeonsun Park, Chunghyeon Lee, Chihyun Han, Uisu Hwang, Seokhwan Jeong

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

Torque observations in reinforcement learning remain challenging because simulated and measured torque differ in scale, offset, and noise. In this paper, we propose a simple torque observation alignment method for robots with direct-drive (DD) actuators, in which motor current maps linearly to joint torque through a motor-type-specific torque constant K_tau. First, dynamometer calibration identifies K_tau* and corrects the scale mismatch between simulated and real torque. Second, the method uses delta_tau(t) = tau(t) - tau(t-1) as the observation in both domains to eliminate the constant offset instead of using the direct torque tau(t), which carries a domain-dependent bias. Third, Gaussian noise obtained from the dynamometer measurement data is injected during the learning process. To validate the proposed method, we train a teacher-student grasping policy entirely in simulation and deploy the distilled student on a multifingered DD gripper. The deployed policy performs proprioceptive grasping using only joint positions and torque differences. We conduct an ablation study comparing the proposed method with alternative alignment variants on nine in-distribution (ID) objects. The proposed method achieves 100% grasp success. These results demonstrate that the proposed alignment method improves the robustness of zero-shot policy transfer on the DD gripper against real-world torque-observation mismatches.

### 论文解读

#### 摘要翻译
强化学习中的扭矩观测存在尺度、偏移和噪声差异。本文为直接驱动（DD）执行器提出扭矩对齐方法：测功机标定电机扭矩常数以修正尺度；仿真和真实策略都使用相邻扭矩差以消除恒定偏移；训练时注入由测量数据估计的高斯噪声。作者完全在仿真中训练教师—学生抓取策略，将仅使用关节位置和扭矩差的学生部署到多指 DD 夹爪。九种分布内物体上达到 100% 抓取成功，说明该方法有助于降低真实扭矩观测失配对零样本迁移的影响。

#### 方法动机分析
仿真策略看到的扭矩可能与实机不同：规格书扭矩常数不准确，绝对电流读数含偏置，传感噪声也未必匹配。策略若依赖这些域特有的绝对数值，直接部署容易失败。作者利用 DD 电机电流与扭矩近似线性的特点，以物理标定修正比例、差分抵消恒定偏移，并按实测误差匹配噪声，避免另训复杂域映射。此假设适用于可校准的 DD 执行器；快速变化偏置及持续负载信息仍是边界。

#### 方法设计详解
先用旋转测功机标定每类电机的有效扭矩常数 $K_\tau^*$，并用测量回归 RMSE 设定高斯噪声。仿真与实机的扭矩关系用尺度、偏置和噪声近似表示；策略观测采用 $\Delta\tau_t=\tau_t-\tau_{t-1}$，恒定基线因相减而消去。随后在 Isaac Sim/Isaac Lab 中训练教师—学生策略：教师为 MLP，利用物体和指尖状态等特权信息通过 PPO 学会抓取；学生为 Transformer，仅输入九维关节位置与九维扭矩差，输出九维相对关节位置变化。学生由教师示范进行行为克隆，部署时以 20 Hz 控制夹爪。系统以抓持高度和水平偏差奖励抓取，并惩罚动作幅值与变化率；仿真随机化质量、摩擦、执行器增益、初始姿态和外部扰动。真实协议依次经过初始状态、抓取、抬升和保持，物体需保持至 10 秒回合结束。训练使用 9,000 个并行仿真环境；教师训练 2,000 次 PPO 迭代，学生用 90,000 条示范轨迹训练 2,000 epochs。学生上下文长度为 30，Transformer 为 6 层、512 维嵌入、8 个注意力头。

#### 方法对比分析
相比直接输入绝对扭矩，本文把误差拆为三项并分别处理：测功机校准改正尺度，时间差分削弱偏置，测量驱动的噪声注入匹配随机误差。其创新是轻量、可解释的物理对齐与标准教师—学生训练组合，而不是新的强化学习算法。适用于具备线性电流—扭矩关系、可完成电机标定的 DD 多指抓取；对非线性较强的减速执行器或需要感知稳态力的任务，尚无证据保证有效。

#### 实验分析（精简版）
真实试验覆盖九种 ID 和 12 种 OOD 物体，每物体进行 10 次抓取。完整方法在 ID 上为 90/90（100.0%），OOD 成功率 98.3%，21 种物体总体 99.0%。消融中，仅位置输入成功率为 15.6%，仅加噪声但未校准尺度、仍使用绝对扭矩为 10.0%；三项机制齐备为 100.0%，显示尺度、偏置与噪声对齐具有互补性。网球和灯泡的 OOD 成功率各为 90%。实验支持固定基座夹爪上的抓取—抬升—保持迁移；温度/电机个体变化及机械臂安装场景未测试，差分也不直接保留持续扭矩大小。

#### 实用指南
复现需对各电机测量并标定有效扭矩常数，以回归残差估噪声，再保证仿真与实机采用一致关节顺序、单位、差分和 20 Hz 控制。论文给出主要网络和 PPO/行为克隆参数及随机化范围，但未完整报告软件版本、计算资源和推理延迟。文中未提供公开代码、原始数据或模型权重链接，仅提到补充视频。换用电机需重新标定；换夹爪或任务则需调整观测/动作映射并重新训练，还应评估稳态负载不可见的影响。

#### 总结
核心思想：标定差分对齐扭矩

1. 测功机校准电机扭矩尺度和噪声。
2. 用扭矩时间差分抵消恒定偏置。
3. 仿真注入实测噪声并随机化物理条件。
4. 蒸馏出只依赖关节位置与扭矩差的学生策略。
5. 零样本部署并检验抓取、抬升与保持。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.29031v1)
- [arXiv](https://arxiv.org/abs/2609.29031v1)

---

<a id='2609.28976v1'></a>
## [ReVNM: Learning-Based Visual Navigation from a Remote Camera](https://arxiv.org/abs/2609.28976v1)

**Authors:** Michikuni Eguchi, Kohei Honda, Masafumi Endo, Yasuhiro Yoshimura, Ryo Yonetani

**Published:** 2026-09-24

**Categories:** cs.RO

**Abstract:**

Visual Navigation Models (VNMs) enable robots to navigate from egocentric visual observations without geometric localization and planning, but long-range navigation still requires pre-built maps. This paper presents the Remote Visual Navigation Model (ReVNM), which uses a single remote surveillance camera to serve as both an observation source and an implicit environmental map for visual navigation. While the use of remote cameras could eliminate the need for pre-built maps as well as onboard vision processing, their limited field of view instead of egocentric observations makes it hard to achieve collision-free navigation. The lack of existing data with diverse remote viewpoints, which are crucial for training robust VNMs, further complicates the challenge. In this work, we propose a learning-by-synthesis approach to address this two-fold challenge. Our ReVNM extends a state-of-the-art VNM architecture with an exocentric-to-egocentric (exo2ego) module that predicts an egocentric depth observation from remote-camera observations. This helps the VNM to plan a path while considering obstacles in front of the robot. Trained only on randomly generated worlds with diverse obstacle layouts and camera viewpoints, ReVNM can generalize well to real robot navigation without additional fine-tuning. Experiments in both simulation and real-world environments confirmed the effectiveness of the proposed approach.

### 论文解读

#### 摘要翻译
视觉导航模型（VNM）可依据第一视角图像导航，但长距离任务通常需要预建地图。ReVNM 使用一台远程监控相机，同时作为观测源和隐式环境地图，从而免去预建地图及机载视觉处理。远程视角受限且与机器人视角不同，障碍物可能遮住机器人附近地面；同时缺少多样远程视角训练数据。作者提出“通过合成进行学习”：增加外视角到自中心视角（exo2ego）模块，从远程图像预测机器人第一视角深度。模型只在含随机障碍布局与相机视角的合成环境训练，无需微调即可迁移至真实机器人；仿真及实机实验验证了有效性。

#### 方法动机分析
远程相机降低了机器人端传感与算力要求，却不能直接看清机器人周围的局部几何。核心假设是，利用仿真可得的机器人位姿和真实自中心深度监督，模型能够从外中心观测合成足以规划的局部深度表示。这样既处理视角差异，也能以随机场景和相机布局弥补实测数据不足。

#### 方法设计详解
输入为固定 RGB 相机图像、机器人在图像中的位置与相对朝向，以及图像中的目标像素。RGB 经 Depth Anything v3 转成外中心深度；系统截取机器人周围区域，将深度和位姿送入 DiT-S/2 条件扩散模型，生成 (32\times32) 自中心深度图，再缩放至 (224\times224)。扩散网络用仿真自中心深度作像素级 MSE 监督，推理采用 8 步 DDIM。导航策略融合长度为 5 的外中心与合成自中心深度历史、机器人状态和目标，使用 ResNet-34 特征骨干及 ACT 风格 CVAE 预测未来 5 个机器人坐标系中的米制航点。训练损失由航点 L1 误差和潜变量 KL 正则组成。航点交给共享的 LiDAR-MPPI 控制器跟踪和避障。训练在 Gazebo 随机障碍环境中收集约 5 万片段、100 万样本，并加入约 5000 个 DAgger 启发的恢复片段。策略训练 20 epochs、AdamW、学习率 (2\times10^{-4})、batch size 512；视角合成模块训练 30 epochs。

#### 方法对比分析
NoMaD 等 VNM 依赖机器人第一视角传感，并常需地图；ReVNM 将感知移至远程相机，用合成自中心深度弥补近身遮挡。相较 IBVS 直接在远程图像中规划，它学习从远程观测到局部航点的映射。关键贡献是视角合成与随机化合成训练，仍假设机器人能被相机定位且未离开视野。

#### 实验分析（精简版）
仿真每个场景进行 250 次试验，成功要求 60 秒内到达且无碰撞。ReVNM 在 Random Pillar、Book Store、Warehouse 的成功率为 88±3%、88±5%、70±6%；Warehouse 中 IBVS 为 52±7%，NoMaD 微调后为 22±4%。移除 exo2ego 后 Warehouse 降至 26±8%；完整模型去掉恢复数据则为 43±6%。实机 Kachaka 的 Wall 场景成功率为 5/5，IBVS 为 0/5，无 exo2ego 变体为 2/5；Forest 场景完整模型为 19/25。结果支持遮挡环境的价值，但实机试验有限，定位使用 AprilTag。

#### 实用指南
复现需构造随机障碍和相机视角，采集专家及恢复轨迹，并以仿真第一视角深度监督扩散模块；部署还需机器人定位、深度估计和 LiDAR-MPPI 控制。实机位姿由 AprilTag 检测，仿真使用真值；推理约需 30 ms 生成深度、0.1 s 预测航点。论文未提供 ReVNM 代码或数据集链接，仅提及可用的 aws-robotics Warehouse 环境。迁移到新相机或机器人需校准像素定位、姿态和深度/坐标约定。

#### 总结
核心思想：远程视角合成机器人局部深度。
1. 从远程 RGB 提取深度并定位机器人与目标。
2. 用位姿条件扩散补出自中心深度。
3. 融合双视角历史预测多步局部航点。
4. 由 LiDAR-MPPI 跟踪，并以恢复示范增强鲁棒性。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.28976v1)
- [arXiv](https://arxiv.org/abs/2609.28976v1)

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
感知运动让类人机器人能通过结构化地形，但在高度不连续环境中跑酷仍有挑战。稀疏落脚点和窄支撑面要求精确选点、利用视觉并保持双脚交替。本文提出仅使用机载深度的跑酷框架，以显著性先验和门控记忆跨帧保留有用深度特征，并用交替损失改进对称性正则。仿真与实机实验显示，该方法提高复杂地形上的通过成功率和落脚精度。

#### 方法动机分析
小平台与窄梁对落脚误差敏感，单帧深度视野有限，下一个踏点可能暂时不可见；直接堆叠历史又会带入冗余。作者假设深度图几何边缘可提示地形线索，历史相对当前越有价值就越应被利用。单纯镜像对称还可能使双腿同步、形成双脚一起跳，因此加入交替约束，鼓励运动中左右脚错相。

#### 方法设计详解
相机图像从 480×270 缩至 64×36，再裁取中央下方 32×18 区域。策略融合角速度、重力方向、速度指令、29 维关节位置与速度、上一动作，以及 8 帧本体感觉和 4 帧深度历史。共享编码器将当前及历史深度映射为潜变量；相邻行深度差经纵向加权得到显著图，其均值概括每帧几何线索。系统计算历史相对当前的显著性差，以 MLP 结合历史潜变量产生 softmax 权重，再把加权历史残差加入当前潜变量。可学习 sigmoid 门控控制修正幅度，保留当前视角并利用此前看到的踏点补充信息。PPO 的非对称 Actor-Critic 由特权状态训练 Critic，Actor 输出 29 个关节目标，经 PD 转为力矩；奖励包括速度跟踪、能耗与动作变化、安全项及 AMP 自然动作先验。镜像损失维持结构一致；速度超过 0.25 时，交替损失惩罚双腿动作余弦相似度的正值。训练使用 Isaac Sim/Isaac Lab、2048 个并行环境和 RTX 4090，最多 50,000 次迭代；Jetson Orin NX 上策略以 50 Hz 推理，深度相机为 60 Hz。

#### 方法对比分析
普通历史拼接让策略自行筛选帧；本文按几何显著性加权，再以残差门控限制记忆影响，针对局部踏点短时不可见。交替损失则在镜像对称之外明确削弱双腿同相动作。该方法无需独立足点检测器，适合局部深度反馈下的快速跑酷；需规划交错落脚或急转向的场景仍受局部视野和缺少高层规划限制。

#### 实验分析（精简版）
仿真在盒子、木桩、横梁、楔形和梯形地形上与 Hiking in the Wild 比较。本文成功率依次为 98.1%、88.1%、91.8%、88.4%、97.1%，基线为 79.4%、70.7%、82.2%、83.5%、64.2%；平均成功率提高 16.7%，落脚精度提高 7.8%。消融的平均成功率/落脚精度从历史均匀聚合的 52.12%/70.80%，到加入显著性后的 88.14%/90.63%，最终门控方案为 98.27%/91.10%。交替损失使成功率从 85.0%升至96.7%，双支撑比例从0.806降至0.308。实机每种地形试验10次，盒子、横梁、梯形成功率超过80%，木桩和楔形约70–80%；试验规模有限，交错桩布局仍较困难。

#### 实用指南
复现起点包括 Isaac Lab、2048 并行环境、深度裁剪、历史长度及部署频率；论文报告学习率 1e-3、对称损失系数 10.0。迁移需调整关节映射、相机视场与裁剪、控制周期和动力学参数，并重新训练评估。作者提供项目主页，但未明确说明代码、模型权重或训练数据已公开。

#### 总结
核心思想：显著性门控记忆助力跑酷
1. 编码当前和历史深度，提取几何显著性。
2. 加权历史信息，以门控残差补全当前表征。
3. PPO 生成关节目标，PD 控制器执行动作。
4. 结合镜像约束与交替损失，稳定步态。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.28960v1)
- [arXiv](https://arxiv.org/abs/2609.28960v1)

---

