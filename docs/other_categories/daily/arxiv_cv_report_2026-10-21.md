time: 20261021

# Arxiv Computer Vision Papers - 2026-10-21

## Executive Summary

## 2026-10-21 arXiv 机器人与视觉论文速览

本期入选10篇，主题集中在让机器人从有限观测中补全空间状态，并把感知转化为可执行、可验证的动作。多篇工作把时序与记忆显式纳入系统：EchoNav聚合历史视觉证据构建带置信度的BEV占据图；EventSync以事件完成证据更新操作记忆，支持长时程重复任务；触觉上下文学习则把触觉、视觉和动作片段组织成演示，让具身任务代理完成精细操作。它们共同反映出，单帧感知或静态提示难以支撑持续交互，状态更新机制本身正在成为系统设计重点。

另一条线索是以机器人实际约束评估方法。SynCalib通过可控合成打破LiDAR-相机标定数据中的捷径；头戴相机UMI研究发现，移动头相机可为低成本示教和观察提供有竞争力的视角；Jetson深度基准则把准确率、帧率、功耗和许可条件放进同一部署选择框架。PanoGraphGS尝试从高斯表示同时构建全景分割与层级场景图。安全与控制方面，眼科RCM机器人用权威感知规划处理约束收紧时的运动退化，Hamilton–Jacobi安全证书工作通过枚举引导训练改善神经安全证书的验证效率。XoNav把步态识别与个性化髋部触觉提示结合，并展示自主导航可行性。

建议优先精读EchoNav、SynCalib和EventSync：分别代表时序空间补全、合成数据控制变量，以及面向重复操作的事件驱动记忆更新。若关注机器人部署，可接着读Jetson深度基准与头相机UMI；关注可靠控制和形式化安全，则读RCM规划与HJ证书。评估这些结果时应区分实机验证、仿真与可行性演示：例如EchoNav包含60次轮式机器人试验，XoNav的自主导航部分是小规模可行性研究，而多项方法仍依赖受限场景或特定硬件条件。

---

## Table of Contents

1. [EchoNav: Evidential BEV Completion from Historical Observations for Visual Navigation](#2610.27616v1)
2. [SynCalib: Breaking Dataset Shortcuts in LiDAR-Camera Calibration via Extrinsic-Controllable Synthesis](#2610.27547v1)
3. [PanoGraphGS: Unifying Panoptic Segmentation and Hierarchical 3D Scene Graphs with Gaussian Splatting](#2610.27696v1)
4. [A Head Camera Is (Almost) All You Need: An Electronics-Free UMI Glove Labelled and Observed by One Moving Camera](#2610.27622v1)
5. [Which Depth Estimator for Jetson AGX Orin? A 2026 Benchmark of Monocular and Stereo Methods](#2610.27199v1)
6. [EventSync: Event-Driven Memory Updates for History-Dependent Manipulation](#2610.27917v1)
7. [High-Precision RCM Motion for Ophthalmic Robotics via Hierarchical QP Control and Authority-Guided Recovery Planning](#2610.27905v1)
8. [Enumeration-Guided Training of Neural Hamilton-Jacobi Safety Certificates](#2610.27870v1)
9. [Embodied Task Agents Can Perform Fine Manipulation through Tactile In-Context Learning](#2610.27613v1)
10. [XoNav: An Exoskeleton as a Personalized Haptic Interface for Assisted Navigation](#2610.27271v1)

---

## Papers

<a id='2610.27616v1'></a>
## [EchoNav: Evidential BEV Completion from Historical Observations for Visual Navigation](https://arxiv.org/abs/2610.27616v1)

**Authors:** Diwen Liu, Yiyang Shao, Borui Wang, Chi Zhang, Yunhui Liu

**Published:** 2026-10-20

**Categories:** cs.RO

**Abstract:**

Visual navigation in cluttered environments requires robots to recover previously observed structure and infer obstacles and free space that remain occluded. We present a framework that combines temporal aggregation and occupancy completion in an evidential bird's-eye-view (BEV) representation predicted from short RGB sequences. A decoder adapts frozen VGGT-Omega features and combines historical residual proposals using cellwise weights and a learned no-update option. The BEV predictor produces a local map in the latest ego frame without depth or camera-pose inputs or explicit geometric warping. The evidential model treats observed free space separately and predicts occupancy with associated confidence across the remaining supported region. We train the predictor on simulated RGB sequences paired with complete and visibility-masked occupancy annotations. Our experiments evaluate navigation in simulation and on a wheeled robot, with additional demonstrations on quadruped and humanoid robots. In simulated sequential two-goal navigation, our system achieves 89.6% success, outperforming the strongest evaluated baseline by 32.5 percentage points. Across 60 wheeled-robot trials in 20 scenes, it achieves 73.3% success, compared with 33.3% for the strongest evaluated baseline.

### 论文解读

#### 摘要翻译
EchoNav研究机器人如何利用先前看到、如今被遮挡的空间信息进行视觉导航。它从短时间内的RGB图像序列预测以机器人当前位置为中心的鸟瞰占据图，同时估计哪些区域确实被观察过、哪些结构属于推断，以及预测有多可靠。系统不需要深度图、相机位姿输入或显式几何配准。作者在模拟导航和轮式机器人上进行实验，并展示了四足与人形机器人上的定性结果。

#### 方法动机分析
机器人只看当前图像时，家具和墙体会遮住通路，局部规划器可能把已见区域当成全部环境，造成重复探索或误判。直接把每帧地图用位姿拼接起来，又会受位姿误差影响。EchoNav的核心假设是：模型可以从多帧视觉中学习各历史视图对当前局部地图的贡献，而且应该允许某个位置不采用历史更新；对于遮挡区域，规划系统还需要知道预测是否有足够空间支持。

#### 方法设计详解
输入是按时间排列的RGB图像。冻结的VGGT-Omega编码器提取图像特征，轻量适配器构造多尺度特征金字塔，跨帧注意力则把图像窗口压成各帧的上下文记忆。模型先以最新帧解码出一个共同的BEV锚点，再让每个历史帧从这个锚点提出残差。逐个BEV格点计算的softmax权重决定各历史残差能改动多少，并加入“保持原样”的候选；这样模型可以只在特定空间位置吸收历史信息。输出网格为256×256，范围10×10米。

占据预测分成观测与推断两部分：一个gate标记已观察到的空闲区域；其余受支持区域通过Beta分布表达占据证据，并另行预测空间支持。置信度由支持范围、占据与空闲的倾向及分布方差共同构成，作者明确说明它不是校准过的碰撞概率。训练标签区分可见空闲、可见障碍、遮挡空闲和遮挡障碍，避免用单一类别损失掩盖不同证据来源。规划器根据占据概率、离障碍距离、机器人外形和可靠性设置行驶代价；未知格保留未知状态并施加有限代价。机器人只走一小段，获取新图像后再更新地图和重规划。

#### 方法对比分析
本方法的区别在于，它与逐帧预测后再做几何配准的方案相比，直接从图像历史解码统一地图，减少对深度和位姿估计的依赖。它也把感知和运动规划分开：地图头专注描述占据与证据，机器人外形、间隙和规划算法仍可按平台调整。代价是模型必须从训练数据中学会跨视角空间关系，训练标注和真实环境差异可能影响遮挡补全。

#### 实验分析（精简版）
作者使用约53.5万个室内训练序列（来自1,000个场景）和约10万个CARLA序列，训练输入约为十帧；测试包含1,200条跨场景路线。十帧历史地图上，置信度过滤后的整体mIoU为68.32%，对比深度辅助OccAnt-RGBD的55.82%；但completion区域的coverage为35.03%，体现准确率与覆盖率的取舍。模拟双目标导航在第二目标出现后成功率为89.6%。实体轮式机器人在20个场景进行60次匹配试验，成功44次（73.3%），对照X-NavDP为33.3%。试验数量有限，且四足和人形结果只作定性展示。

#### 实用指南
适合有RGB相机、采用局部占据图进行规划的移动机器人。复现时应保留“未知”状态，并分别调节机器人外形膨胀、未知区惩罚和置信门槛；文中置信分数不是安全概率。作者冻结VGGT-Omega，仅训练适配器、时序解码器和预测头。论文未给出推理延迟，部署前需在目标硬件测量，并验证真实环境中的动态障碍和域偏移。

#### 总结
核心思想：让视觉历史补全局部地图。

1. 编码连续RGB图像，得到跨帧视觉记忆。
2. 以最新视角建立BEV锚点，按格点选择历史残差或不更新。
3. 分别预测观测空闲、推断占据、空间支持和不确定性。
4. 将地图与机器人外形代价交给规划器，短程执行后重新观察。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27616v1)
- [arXiv](https://arxiv.org/abs/2610.27616v1)

---

<a id='2610.27547v1'></a>
## [SynCalib: Breaking Dataset Shortcuts in LiDAR-Camera Calibration via Extrinsic-Controllable Synthesis](https://arxiv.org/abs/2610.27547v1)

**Authors:** Huihua Zhu, Shuyang Zhang, Dongqing Chen, Zhuo Yu, Jianhao Jiao, Yilong Zhu, Siyu Teng, Xiaopin Zhong, Zongze Wu

**Published:** 2026-10-20

**Categories:** cs.CV, cs.RO

**Abstract:**

Online LiDAR-camera calibration remains a fundamental yet challenging task in the field of robotics. Recently, deep learning-based data-driven approaches have emerged, offering high-precision, targetless calibration capabilities. However, existing methods often suffer from a critical dataset shortcut phenomenon due to the lack of extrinsic diversity in their training data. Networks tend to overfit to fixed nominal extrinsic configurations, leading to severe performance degradation and poor generalization in real-world deployments. To address this limitation, we propose a highly generalizable data generation framework tailored for training LiDAR-camera calibration networks. Our framework leverages existing omnidirectional image datasets annotated with depth ground truth to synthesize multi-modal calibration data under rigorous extrinsic constraints. Extensive experiments demonstrate that the data generated by our pipeline significantly enhances the generalization and robustness of typical learning-based calibration networks. Furthermore, we conduct comprehensive real-world evaluations across various camera-LiDAR configurations, validating the zero-shot transfer capability of models trained exclusively on our synthetic data. To facilitate future research and community use, the code for data generation, the pre-trained models, and online benchmark datasets will be made publicly available.

### 论文解读

#### 摘要翻译
SynCalib关注学习式LiDAR-相机在线标定在真实设备上容易失效的问题。作者认为，常见训练集的相机和雷达外参过于固定，使网络学到投影分布捷径，而非稳健的跨模态几何对应。为此，论文从带深度标注的全景图像合成具有可控外参的多模态训练数据，并用实体传感器进行零样本验证。

#### 方法动机分析
机器人运行时，振动和热变化会让传感器安装外参偏离出厂值，重新进行靶标标定成本较高。现有学习方法通常在输入点云上加随机扰动，但训练样本的真实几何与标注仍围绕少数固定配置，网络可能只记住这些配置下的投影模式。本文的核心假设是，如果数据生成阶段就覆盖多样、严格受控的相机—雷达外参，模型便更难依赖固定的空间先验。

#### 方法设计详解
SynCalib的合成流程使用TartanGround中的全景RGB和深度图。第一步将六面全景转换为等距柱状图；第二步根据深度和雷达扫描pattern合成三维点云，并按目标相机内参从全景中裁出透视图。第三步利用两张全景之间已知的相对位姿构造传感器对：缩放场景深度以匹配目标平移长度，旋转相机方向以匹配平移方向，再求出满足完整目标外参的旋转。平移约束还留有绕基线旋转的一维自由度，方法均匀采样该角度来增加合法视角。生成后先用双向深度一致性筛选两视图的共视区域，再检查透视裁剪后点云与图像的重叠率，并保留含足够深度边缘的样本。论文以CMRNext训练300个epoch、CalibNet训练30个epoch。

该方法调整数据生成方式，不改变下游标定网络结构。训练时点云仍按标称工厂外参投影，而监督目标对应生成时变化的实际外参，迫使网络学习图像与点云之间的几何错位。作者对比了几何匹配式CMRNext与直接回归式CalibNet，并采用旋转误差、平移误差和投影端点误差评估。

#### 方法对比分析
本方法的创新贡献在于把多样性从训练时的输入扰动提升为数据本身的几何外参变化，因此可与已有网络兼容。相对只做输入增强的传统方案，它明确构造具有新外参真值的数据对；相比修改模型结构来缓解捷径，它提供一个可复用的数据生成管线。适用前提是能获得全景图像及可信深度，并且合成几何与目标硬件的视场、扫描pattern足够接近。

#### 实验分析（精简版）
作者从20个结构化场景合成32,000对训练样本和8,000对测试样本，CMRNext训练300个epoch、CalibNet训练30个epoch。实体测试覆盖Mid-360和RS-LiDAR-32，每个传感器配置包含三个外参状态、每状态40对数据。在Mid-360的两个实体错位状态上，SynCalib训练的CMRNext中位平移误差为11.97厘米和9.98厘米，传统训练为21.47厘米和17.55厘米。三阶段合成推理达到1.06厘米、0.24度的中位误差。零样本实体测试仍有明显平移误差，部分RS-LiDAR-32状态也未全面优于传统方法，作者将其与合成和真实基线分布不匹配联系起来。

#### 实用指南
适合需要标定相机与LiDAR外参的机器人和自动驾驶团队。复现应重点核对坐标变换方向、深度缩放、扫描pattern、双向可见性筛选以及输入与监督外参定义。代码、预训练模型和在线基准计划公开；论文数据由20个结构化场景构成，场景多样性有限。迁移到目标硬件后应分别报告旋转、平移和像素投影误差，并针对稀疏雷达与低重叠视场单独验证。

#### 总结
核心思想：让合成数据覆盖真实外参变化。

1. 从全景RGB和深度生成透视图与雷达点云。
2. 按目标外参几何约束合成多样传感器对。
3. 筛除共视不足或几何边缘稀少的样本。
4. 用多样外参数据训练现有标定网络并做实体零样本评估。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27547v1)
- [arXiv](https://arxiv.org/abs/2610.27547v1)

---

<a id='2610.27696v1'></a>
## [PanoGraphGS: Unifying Panoptic Segmentation and Hierarchical 3D Scene Graphs with Gaussian Splatting](https://arxiv.org/abs/2610.27696v1)

**Authors:** Jinghan Gao, Siyu Wang, Dongyu Feng, Yiyuan Pan, Rong Xiong

**Published:** 2026-10-20

**Categories:** cs.CV, cs.RO

**Abstract:**

3D scene graphs support language-guided navigation by organizing physical entities and their containment and spatial relations into hierarchical maps. Building such structured maps requires a scene representation that consistently distinguishes semantic regions and object instances across viewpoints. To this end, we present PanoGraphGS, a unified framework that integrates panoptic 3D Gaussian representation learning with hierarchical 3D scene graph construction. The framework learns semantic and instance features on shared Gaussian geometry. Prototype-based semantic learning and association-based instance learning produce consistent panoptic predictions across viewpoints. The learned representation supports novel-view panoptic segmentation and direct Gaussian-level panoptic decoding. A semantic-aware scene graph module further recovers a floor-room-stuff-object hierarchy with containment and spatial relations. We further release NaviGS, a real-world dataset for panoramic scene understanding. The dataset provides 3DGS, mesh, RGB, semantic, and depth data. For 3D scene graph identification, PanoGraphGS outperforms the strongest evaluated baseline in object-node accuracy by more than 50\% points on average across Replica and NaviGS. PanoGraphGS also achieves state-of-the-art performance among the evaluated methods in 2D panoptic segmentation. These results demonstrate the effectiveness of PanoGraphGS in both hierarchical 3D scene understanding and 2D panoptic segmentation. The project website is available at: https://github.com/gaojh0333/PanoGraphGS.

### 论文解读

#### 摘要翻译
PanoGraphGS把全景分割和层次化三维场景图结合起来。它在共享的3D Gaussian几何上学习语义类别和持久实例身份，利用多视角RGB与分割掩码预测新视角的全景分割，并进一步构建楼层、房间、结构物和对象组成的场景层次。作者还发布真实全景数据集NaviGS，用于场景理解评估。

#### 方法动机分析
语言导航需要的不只是识别床或椅子，还要知道它属于哪间房、位于哪层。传统场景图往往依赖深度图构造几何，深度误差会传导到物体范围和房间划分；分割与场景图分开处理，又需要额外把两种表示对应起来。论文的假设是，若语义、实例身份与几何共享同一Gaussian表示，就能从不同视角保持对象身份一致，并直接支持层次关系推理。

#### 方法设计详解
输入为带相机位姿的多视角RGB图像及SAM3生成的语义、实例掩码。每个Gaussian除外观与空间参数外，附带64维语义特征和实例特征；可见性感知渲染将图像、语义和实例特征投影到像素。语义分支使用共享残差投影和类别原型，使像素特征与Gaussian特征在同一空间分类。实例分支为每个对象维护全局原型，把每个视角中局部实例掩码区域与全局身份计算匹配代价，再用匈牙利算法做一对一关联，解决同一对象在不同视角获得不同伪标签的问题。两条分支通过分类、区域对比、原型对齐和置信加权进行联合学习，以降低遮挡和物体边界附近噪声标签的影响。

得到全景Gaussian后，系统直接解码stuff语义和对象类别/身份。场景图构建先按语义与几何抽象实体；地面、天花板用于识别楼层，墙壁、门窗等结构线索划分房间。随后渲染房间内RGB视图，借助CLIP推断房间功能类别；对象依据其三维Gaussian支撑点落入各房间包围盒的数量分配归属。最终形成楼层—房间—结构物—对象的包含关系和空间关系。训练按场景独立优化，每场景30,000次Adam迭代，学习率2.5×10⁻³，实例原型数为256。

#### 方法对比分析
本方法的关键区别是将跨视角全景分割、Gaussian级实体和层次场景图建立在同一表示中。相较只做对象聚类的场景图方法，它显式维护跨视角对象身份及楼层、房间层次；相较先分割后组图的流水线，不需另训练图预测器。它仍依赖带位姿RGB和外部分割掩码，当前房间划分主要利用结构语义与几何规则。

#### 实验分析（精简版）
在真实NaviGS数据上，楼层、房间、对象准确率分别为100%、100%和98.36%，对象节点准确率比最强已测基线高逾50个百分点。全景分割mIoU/mBIoU为83.19/64.39，对比Unified-Lift的78.48/57.84。NaviGS由鱼眼相机与手持LiDAR辅助采集，测试选择多房间场景；当前方法离线处理，开放式厨房等缺少清晰边界的空间仍难以划分，评估也不等同于在线导航成功率。

#### 实用指南
适用于需要持久对象身份、房间级地图或语言目标定位的室内机器人。复现需准备多视角位姿、Gaussian场景和质量较好的语义/实例掩码，并按场景优化模型。论文公开NaviGS与项目代码；部署到在线系统时还要处理持续观测、动态物体和开放空间边界。自动驾驶中的道路层级和移动对象关系需要重新定义，本文未测试此类迁移。

#### 总结
核心思想：以共享Gaussian维持身份并组织场景。

1. 从多视角RGB和掩码学习Gaussian语义与实例特征。
2. 将视角局部实例关联到持久全局身份。
3. 从结构语义和几何推断楼层、房间及对象归属。
4. 输出可用于语言空间查询的层次场景图。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27696v1)
- [arXiv](https://arxiv.org/abs/2610.27696v1)

---

<a id='2610.27622v1'></a>
## [A Head Camera Is (Almost) All You Need: An Electronics-Free UMI Glove Labelled and Observed by One Moving Camera](https://arxiv.org/abs/2610.27622v1)

**Authors:** Yosub Shin, Yuan Liu, Igor Molybog

**Published:** 2026-10-20

**Categories:** cs.RO, cs.AI

**Abstract:**

Handheld demonstration devices decouple data collection from the robot, but conventional designs burden the hand with a wrist camera, cables and a pose tracker (SLAM, an IMU, a VR controller, or a fixed external camera that the collector's body occludes), and burden the pipeline with as many data streams to timestamp, synchronise and cross-calibrate. We show that a single egocentric camera (USD 122) worn on the collector's head replaces the wrist camera, the pose tracker and their separate streams with one. It reads printed fiducials on an electronics-free glove (USD 22) against paper tiles taped around the workspace, giving the 6-DoF gripper pose and the jaw width from the same frames the policy will see; at deployment the same camera rides a low-cost (USD 120) arm beside the robot that mimics the collector's head from the robot's joint state. Against the robot's encoders the labels are accurate to 6.5 mm and 1.6 degrees median with 93-100% frame coverage, and a policy trained on them instead of encoders pays 3% in validation loss. Adding 194 glove demonstrations to 48 teleoperated episodes lifts rollout success from 10/25 to 17/25, level with 172 teleoperated episodes alone (18/25), and the full co-trained head-camera policy succeeds in 20/25 against 15/25 for the wrist-camera recipe, whose policies drop the object when the goal leaves their view, and 12/25 for a fixed camera.

### 论文解读

#### 摘要翻译
本文提出一种简化的机器人示教采集方案：操作者头上只戴一台相机，手持不含电子元件的夹爪仿形手套。相机通过手套上的标记和工作区纸质标签，同时估计夹爪位姿、开度和周围画面；机器人执行时，同一相机安装在旁边的小型机械臂上，近似跟随夹爪与操作者视角。作者报告标注精度，并将采集数据与遥操作示教混合训练策略。

#### 方法动机分析
传统手持示教装置可能需要腕部相机、独立追踪器、线缆和多路同步标定。腕部视角还会随手运动而离开放置目标，固定相机则容易被操作者身体遮挡。论文的核心假设是，头部视角能在抓取与搬运过程中同时保留夹爪和目标；若标记布置得当，一台相机也能提供世界坐标、夹爪位姿与开度标签，从而减少硬件和时间同步负担。

#### 方法设计详解
采集时，操作者佩戴15 Hz头戴相机，使用成本约22美元的被动仿形手套。手套背部的11面ArUco球提供夹爪六自由度姿态，夹爪上的标签尾片估计夹爪开度；工作区的纸质AprilTag tile用于定义世界坐标。系统先从tile估计相机位姿，再通过PnP解出手套相对相机的姿态，组合成夹爪在工作区内的轨迹。tile短暂不可见时，算法在速度限制下沿用上一可信位姿；跟踪中断过久则切分数据段。

部署时，同一相机改装在机器人旁的六轴positioner上。positioner根据机器人关节编码器跟随夹爪，并用ridge regression预测操作者头部相对手部的视角变化，让训练与执行尽可能共享视角。策略采用ACT和LeRobot，训练步数为50k。实验比较单独遥操作、加入手套示教的混合数据，以及头戴、腕部鱼眼和固定相机等观察方式。

#### 方法对比分析
本方法的主要创新是把视角与追踪合并到单路头戴相机：腕部标记负责动作标签，工作区tile负责坐标定位，避免单独的SLAM/IMU追踪器。相较腕部相机，头戴视角更容易同时看到夹爪和放置目标；相较固定相机，采集者移动时仍能保持夹爪可见。系统仍依赖标签可见，且机器人端需要positioner近似重现采集视角。

#### 实验分析（精简版）
在5,065帧上，夹爪位姿标签的中位误差为6.5毫米、1.55度。加入194条手套示教后，48条遥操作数据的成功数从10/25提高到17/25；全量混合策略为20/25。相机视角比较中，头戴、腕部和固定相机分别为20/25、15/25和12/25，但每组仅25个放置位置，配对检验未达到常用显著性门槛。结果支持低成本数据补充的潜力，尚不足以证明不同用户或任务上稳定领先。

#### 实用指南
适合低成本桌面抓取示教和需要夹爪动作标签的数据采集。需打印与夹爪匹配的手套、标记和tile，并确保相机与tile/手套同时可见。当前positioner有约0.4秒延迟，测试仅一名操作者和单一任务；部分硬件工具已开源，机器人侧标定和数据工具尚未发布。扩展到不同夹爪或双臂时应重新标定坐标关系并评估遮挡与视角延迟。

#### 总结
核心思想：一台头戴相机同时看场景与动作。

1. 用被动标记手套记录夹爪姿态和开度。
2. 用工作区标签将视觉位姿转为统一坐标轨迹。
3. 在机器人旁复用相机并近似重现采集视角。
4. 混合手套示教与遥操作数据训练抓取策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27622v1)
- [arXiv](https://arxiv.org/abs/2610.27622v1)

---

<a id='2610.27199v1'></a>
## [Which Depth Estimator for Jetson AGX Orin? A 2026 Benchmark of Monocular and Stereo Methods](https://arxiv.org/abs/2610.27199v1)

**Authors:** Leaf Jiang

**Published:** 2026-10-19

**Categories:** cs.RO, cs.CV

**Abstract:**

The Jetson AGX Orin is a common computer for robots and vehicles, yet measured accuracy, speed and energy of current depth estimators on it are hard to find. We benchmark nine model families as of October 2026: the monocular Depth Anything V2 and 3, Depth Pro, MoGe-2, Metric3D v2 and UniDepth V2, and the stereo methods libSGM, NVIDIA ESS and Fast-FoundationStereo. Every method runs on the Orin as a TensorRT FP16 engine or native CUDA code at up to three input sizes, up to the full 5.7 MP, and is scored for metric accuracy against Middlebury 2014 ground truth (non-occluded pixels of the 10 scenes whose disparity stays below 256 px), GPU time, energy and licence. Stereo is far more accurate. At 30 fps, NVIDIA's ESS reaches 0.70% AbsRel in 8.6 ms and libSGM 0.73% in 25 ms; with more compute, Fast-FoundationStereo reaches 0.33%. The best monocular models reach 12% with the true focal length (Depth Anything 3 Metric-L in 33 ms, Depth Pro in 803 ms), and 3.8-4.0% only with an oracle per-image scale (MoGe-2-L, UniDepth V2-B, Metric3D v2-L). Stereo error falls as the input resolution grows; no monocular model improves, and the metric scale of most drifts away from the truth outside their training resolution. At depth discontinuities, which AbsRel barely registers, the monocular models scatter 17-46% of the edge points between foreground and background, learned stereo 2-9% and libSGM at full resolution 1%. We give recommendations per frame-rate budget and release all code, per-scene results and engine build scripts.

### 论文解读

#### 摘要翻译
本文在Jetson AGX Orin上比较九类单目和双目深度方法，统一测量误差、GPU耗时、能耗和模型许可，并按帧率预算给出选型建议。实验显示，在所测设备与近距离室内数据中，双目方法的公制深度精度明显高于单目模型；单目模型把输入放大到训练分辨率之外，也没有稳定获益。

#### 方法动机分析
深度论文常在不同设备、输入尺寸、数值精度和真值处理方式下报告结果，难以直接判断哪种方案适合嵌入式机器人，这正是选型的痛点。作者希望给出一份同设备、同数据和同指标的实测基准。其核心假设是，双目视差通过相机几何直接提供尺度，而单目网络依赖训练得到的场景尺度先验；增加像素未必能修正这种先验偏差。

#### 方法设计详解
整体处理流程是：在Jetson AGX Orin 64GB上运行九类方法，包括单目的Depth Anything、Depth Pro、MoGe-2、Metric3D和UniDepth，以及双目的libSGM、NVIDIA ESS和Fast-FoundationStereo；网络以TensorRT FP16引擎计时；当FP16结果不可靠时，使用FP32模型的准确度，并单独核查引擎差异。每种方法在原生尺寸及设备允许的更大尺寸运行。评价使用Middlebury 2014真值，将预测深度换算到公制后计算AbsRel、阈值准确率、覆盖密度、GPU时间和能耗；作者还单独统计物体边界处的飞点。对单目模型，逐图oracle尺度校正只作为分析形状误差的上界，不能视作部署结果。实验采用10个视差低于256像素的室内场景和非遮挡真值，且数据经过完美校正。

#### 方法对比分析
这项工作的贡献是统一硬件与测量协议，并把分辨率、边缘错误、许可和算力预算放在同一选型框架。双目能利用标定视差建立公制尺度；单目则易受场景先验和训练分辨率影响。即使全分辨率libSGM边缘飞点少，也会丢失不少边界像素，因此密集学习式双目和传统匹配器适用条件不同。结果适合指导与该平台、输入和场景相近的部署，不能替代目标传感器上的现场测试。

#### 实验分析（精简版）
在10个近距离室内场景上，30 fps预算下NVIDIA ESS达到0.70% AbsRel、8.6 ms；同预算单目Depth Anything 3 Metric-L为12.34%、33 ms。最高精度的Fast-FoundationStereo达到0.33%，但单帧需要758 ms。边缘处单目方法有17–46%飞点，学习式双目为2–9%；全分辨率libSGM仅1.1%，但只覆盖45%的边缘像素。范围限于完美校正的Middlebury室内数据，且排除了视差超出搜索范围的场景。

#### 实用指南
论文提供基准代码、逐场景结果和引擎构建脚本。复现时应固定Orin软件栈、功耗模式、预处理和计时边界，并确认FP16数值一致性。选择模型时同时检查输入尺寸、真焦距可用性、覆盖率、边缘行为、能耗和许可；部署在不同相机或远距离场景时，应重新测量，不能只按AbsRel排名。

#### 总结
核心思想：用同平台实测选深度方案。

1. 在目标嵌入式设备统一运行候选方法。
2. 按真实标定和真值比较精度、延迟、能耗与边缘误差。
3. 结合帧率、场景范围和许可作部署选择。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27199v1)
- [arXiv](https://arxiv.org/abs/2610.27199v1)

---

<a id='2610.27917v1'></a>
## [EventSync: Event-Driven Memory Updates for History-Dependent Manipulation](https://arxiv.org/abs/2610.27917v1)

**Authors:** Jingjiao Zhao, Dong Jing, Mingyu Cao, Ruochuan Shi, Xiansheng Chen, Pengwei Wang, Zhiwu Lu

**Published:** 2026-10-20

**Categories:** cs.RO

**Abstract:**

History-dependent robotic manipulation requires task context and interaction history, as the current observation alone is insufficient to determine the next action. Hierarchical frameworks use task memory for high-level subgoal reasoning and low-level policies for execution. However, memory updates require accurate completion assessment: similar observations may correspond to different execution phases, and prematurely recording completion can mislead subsequent subgoal reasoning. We present EventSync, a hierarchical framework that coordinates task reasoning and action execution through event-driven memory updates. A vision-language model generates subgoals from task plans and interaction history, while a vision-language-action model executes them. A lightweight Event Monitor combines visual observations with end-effector trajectories to determine whether the current subgoal is complete. The framework records the current subgoal in its completed-subgoal history and triggers subsequent reasoning only after obtaining completion evidence. With visual history in the executor, EventSync achieves 64.5% overall success on RoboMME, exceeding the best baseline reported in the benchmark by 20.0 percentage points. Further analysis shows that this update mechanism helps build a reliable completed-subgoal history. Experiments on four real-robot tasks with different memory requirements validate the framework's effectiveness in real-world robotic manipulation.

### 论文解读

#### 摘要翻译
历史依赖操作不能仅依据当前图像决定下一步。EventSync把任务推理、动作执行和完成监控分开：高层模型根据指令及交互历史生成子目标，VLA执行动作，轻量监视器根据图像和末端轨迹判断操作是否完成。只有获得完成证据后，系统才更新已完成子目标并继续推理。

#### 方法动机分析
固定频率地更新高层记忆容易误判动作进度：同一姿态可能出现在按压过程和回程阶段，若提前记录完成，机器人会跳到下一子目标。作者的核心假设是，把视觉变化与末端运动结合起来评估事件进度，能让记忆更新跟随真实执行阶段，减少过早推进。

#### 方法设计详解
Memory Reasoner读取语言指令、可选演示视频、当前图像、历史图像和已完成子目标，生成任务计划与空间定位的当前子目标。Action Executor依据子目标和机器人观测产生动作块。Event Monitor接收子任务开始及当前画面、末端轨迹和事件条件，输出操作完成概率；对于传送带等环境事件，它还预测两次查询之间新增事件的数量，并累计进度。视觉token由冻结的DINOv2提取，Transformer融合视觉、运动轨迹和事件条件；GRU保存跨时刻状态以预测事件增量。只有完成监控提供证据，子目标才会写入完成历史并触发下一轮高层推理。监视器有87.39M参数，训练30轮、学习率3×10⁻⁴；高层Qwen3-VL-4B使用LoRA微调两轮，动作执行器从π0.5继续微调。

#### 方法对比分析
区别于按动作块数或固定周期更新的层级控制，EventSync把子目标完成检测作为记忆更新的触发条件；区别于让大型语言视觉模型兼任判断，它使用专门的轻量监视器，并显式加入末端轨迹。它更适用于需要计数、顺序记忆、动作阶段判断的操作任务，且依赖可定义的事件和可靠完成标注。

#### 实验分析（精简版）
在16项RoboMME任务上，含执行器视觉历史的EventSync成功率为64.5%，高于基准报告的FrameSamp+Modul 44.5%。摆动计数任务中，提前写入完成历史的回合从64.0%降至10.0%（各50轮）。四项实机任务每种方法各测10次，EventSync分别成功9/10、7/10、9/10和7/10；样本量较小，实机结论仍属初步证据。

#### 实用指南
复现需准备按事件标注的演示和交互数据、对齐末端轨迹、定义完成条件，并为每类子目标训练完成或事件增量监视器。作者的实机评估使用Franka Panda和Robotiq夹爪；动作执行器使用每任务20步训练块、16步推理块。迁移到新机器人时，要检查摄像头视角、控制延迟、事件映射规则和完成标签质量，并分别量化提前更新与漏更新的代价。

#### 总结
核心思想：以执行证据驱动任务记忆更新。

1. 根据任务指令和历史生成当前子目标。
2. 执行动作并用视觉与轨迹估计事件完成。
3. 确认完成后更新历史，再规划后续操作。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27917v1)
- [arXiv](https://arxiv.org/abs/2610.27917v1)

---

<a id='2610.27905v1'></a>
## [High-Precision RCM Motion for Ophthalmic Robotics via Hierarchical QP Control and Authority-Guided Recovery Planning](https://arxiv.org/abs/2610.27905v1)

**Authors:** Huanyu Tian, Adriana Namour, Lingyun Zeng, Zicong Wu, Sebastien Ourselin, Lyndon da Cruz, Christos Bergeles

**Published:** 2026-10-20

**Categories:** cs.RO

**Abstract:**

General-purpose manipulators can position ophthalmic instruments under a software Remote Centre of Motion (RCM). However, at some reachable configurations that are nonsingular for the tip task alone, the RCM and tip-tracking objectives conflict and the servo oscillates. We formalise this failure as a constraint-induced task singularity: the RCM-projected tip Jacobian loses rank while the unprojected tip Jacobian stays well conditioned. Thus, the effective tip authority collapses although the target remains reachable. We present a high-precision RCM servo architecture with built-in singularity recovery: for each commanded target, an authority-guided planner selects a well-conditioned arm configuration that reaches it and plans an RCM-consistent route, executed by a fixed hierarchical-QP controller. On a 200-scenario kinematic benchmark, the complete nominal-and-recovery protocol reaches 128/128 near-singular targets and 71/72 near-joint-limit targets (100% and 99%, respectively), with the software-RCM residual below 0.142 mm. On a physical 6-DOF arm it executes across four RCM entry points, with a worst-case kinematically reconstructed RCM residual of 178 μm and a median tip-regulation error of 32μm, within the 200μm control requirement adopted for the ophthalmic task.

### 论文解读

#### 摘要翻译
眼科机器人必须让器械尖端到达目标，同时保持器械轴始终通过巩膜入口处的远程运动中心（RCM）。作者分析一种受RCM约束诱发的控制奇异现象，并提出分层QP控制器与基于有效控制权的路径恢复规划器，在奇异或关节限位附近选择更易控制的构型。

#### 方法动机分析
一般机械臂可用软件而非机械连杆实现RCM，但末端运动和RCM约束会争用关节速度。某些末端本身可达的姿态，投影到RCM允许运动后却失去有效控制权，QP可能振荡或停滞。作者的假设是，提前衡量RCM投影后的末端控制权，并规划绕开低权威姿态的路径，比仅靠局部零空间调节更能稳定到达。

#### 方法设计详解
整体流程是：低层固定控制器采用两级有界松弛HQP，第一级优先控制RCM横向残差并限制容许松弛，第二级在不损害第一级最优值的条件下跟踪末端速度、平滑关节运动。上层先沿起点到目标的直线逐步推进参考。若有效权威、条件数或关节限位裕度触发风险门控，规划器用多初值受约束逆运动学找出可达且满足RCM及关节约束的候选构型，再选取RCM投影末端雅可比最小奇异值最大的一个。之后在RCM流形上用切向步进生成路径，并通过Newton回投影抑制离散误差；前瞻末端参考和零空间关节引导交给同一个HQP执行。若末端停滞则重新规划。该方法挑选的终点控制权较高，但不保证整条路径都有权威下界。

#### 方法对比分析
相比只在当前构型做零空间调节，本文加入多初值IK和RCM流形路径规划，提供局部控制器缺少的路径前瞻；相比单纯放慢直线参考，规划可以主动绕过低权威区域。适用于高精度RCM点操作和存在关节限位的机械臂。其实体结果度量的是软件RCM和机器人正运动学坐标中的调节表现。

#### 实验分析（精简版）
在固定200个运动学情景中，含一次失败恢复的完整流程到达全部128个奇异情景和72个近限位情景中的71个，最坏软件RCM残差0.142毫米。实体Meca500在四个入口点的47个目标中到达44个（94%），末端误差中位数32微米，重建RCM残差最大178微米。硬件指标由正运动学计算而来，并非独立的眼内解剖测量；实体试验验证实时执行，未单独比较奇异恢复性能。

#### 实用指南
复现需要机械臂正运动学、RCM几何、雅可比和约束一致的坐标系，并在50Hz伺服中运行有界松弛HQP。需标定风险阈值、逆运动学候选搜索、流形路径步长和回投影精度。面向真实手术迁移还要加入碰撞与解剖约束，以独立传感或外部基准验证绝对精度；控制误差满足软件阈值本身并不等同于组织安全。

#### 总结
核心思想：绕开RCM约束下的低控制权路径。

1. 计算RCM投影后的末端有效控制权。
2. 选择可达且高权威的目标构型并生成流形路径。
3. 用分层QP跟踪路径，停滞时重新规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27905v1)
- [arXiv](https://arxiv.org/abs/2610.27905v1)

---

<a id='2610.27870v1'></a>
## [Enumeration-Guided Training of Neural Hamilton-Jacobi Safety Certificates](https://arxiv.org/abs/2610.27870v1)

**Authors:** Mihaela-Larisa Clement, Tosca Pierro, Ezio Bartocci, Luca Marzari

**Published:** 2026-10-20

**Categories:** cs.RO, cs.LG

**Abstract:**

Hamilton-Jacobi (HJ) reachability analysis offers formal safety guarantees for control systems, but training accurate neural certificates remains difficult, particularly when unsafe regions are hard to characterize. We propose a verification-guided training framework that addresses this by using probabilistic enumeration of safety constraints to detect forward-invariance violations, i.e., states where the learned certificate incorrectly classifies unsafe behavior. This produces a compact, targeted set of counterexamples that more precisely captures the failure boundary than inputs generated by standard adversarial attack methods. Crucially, using this representation to fine-tune a pretrained HJ value function yields larger certified regions and reduces certification time across all tested benchmarks. We further introduce a shielding mechanism based on the refined certificates to enforce safety constraints at deployment time. We evaluate the proposed safety filter in a realistic, physics-based Isaac Sim environment and through real-world autonomous racing experiments, achieving zero safety constraint violations compared to a nominal controller and a backup policy based on Linear Quadratic Regulation (LQR). These results demonstrate the effectiveness of the proposed shielding mechanism in ensuring safe operation under high-speed, safety-critical conditions.

### 论文解读

#### 摘要翻译
本文提出一种用于神经Hamilton–Jacobi安全证书的验证引导训练方法。它通过概率枚举定位状态空间中的违规区域，再从中抽取反例微调预训练值函数，并保留形式验证作为最终安全证明。作者还把通过验证的证书用于运行时安全过滤器，在名义动作可能越出安全区域时切换到备份控制器。实验覆盖控制基准、RoboRacer仿真和实体赛车展示。

#### 方法动机分析
HJ可达性可以描述长期安全状态集合，但精确求解高维偏微分方程很困难，因此研究者使用神经网络近似值函数。近似网络本身不保证安全，仍须形式验证。既有流程常用PGD在当前状态附近寻找违规点，局部搜索可能漏掉远处失败区域，使MILP验证器反复发现新反例。本文的假设是，先用区域级探索获得更全面的反例形状，可以减少遗漏并加快收敛；最终形式验证仍不可省略。

#### 方法设计详解
安全证书是神经值函数零次水平集。其必须满足两点：集合中状态符合约束；在备份策略作用下，下一步仍留在集合内。算法从预训练值函数开始，利用bootstrap样本训练随机决策树集成，以当前证书标注违反约束或前向不变性的状态。树将状态空间划分为轴对齐区域，集成结果近似给出违规区域；系统从这些区域抽样，并对候选逐点精确检查，只把真实违规点加入缓存来更新值网络。样本不够时在已有反例邻域补充搜索，若多次更新未取得进展则调用MILP做全域形式验证；验证器返回反例时继续训练，通过时才结束。消融实验的训练最多进行5,000次迭代。

部署时，名义控制器提出动作，盾层计算该动作的一步后继，并检查值函数是否仍不大于零。若安全则放行，否则改用预先验证的饱和LQR备份动作。证书网络为两层、每层32个ReLU单元。理论保证要求备份策略模型与实际系统动态相符；作者指出，备份模型误差可考虑用Lipschitz裕量处理。

#### 方法对比分析
本方法的创新在于以空间区域而不是稀疏局部点作为反例搜索中间表示，提升训练覆盖并减少反复验证；与随机均匀采样相比，违规区域能提供更集中监督。区域枚举不等同于安全证明，正式认证仍由MILP或其他验证器完成。轴对齐区域遇到弯曲边界可能包含大量无效空间，降低采样效率。

#### 实验分析（精简版）
在六维机器人手臂基准上，枚举方案真可行率为69.87%，PGD基线为49.02%；平均墙钟时间由7,955秒降至1,732秒。该评估的真可行率根据10万均匀状态样本估计，结果为三次运行均值。CartPole上枚举证书略小，作者将其归因于弯曲违规边界难以用轴对齐区域表达。RoboRacer仿真五次随机障碍测试中，盾层零碰撞，名义控制器5次均碰撞、备份控制器碰撞4次；平均干预率58.2%。作者另展示实体车在未见赛道绕过六个障碍，但未说明实体重复试验次数。

#### 实用指南
适用于具有可验证动力学近似、明确安全集合和可靠备份控制器的系统。训练需配置区域枚举、反例校验、形式验证器和验证超时；运行时还要校核状态估计、执行延迟及实际动态与认证模型之间的偏差。代码与实验配置可获取。用于自动驾驶时需把感知误差、车辆模型误差和备份控制约束纳入认证，实体演示尚不足以代表统计可靠性。

#### 总结
核心思想：用区域级违规反例训练可验证安全盾。

1. 以神经值函数定义候选安全区域。
2. 枚举可能违规区域并筛出真实反例。
3. 微调证书，周期性调用形式验证器确认安全。
4. 在线拒绝危险名义动作并调用备份控制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27870v1)
- [arXiv](https://arxiv.org/abs/2610.27870v1)

---

<a id='2610.27613v1'></a>
## [Embodied Task Agents Can Perform Fine Manipulation through Tactile In-Context Learning](https://arxiv.org/abs/2610.27613v1)

**Authors:** Yubang Wang, Zezheng Huai, Yifei Zhang, Jinhao Wu, Yitong Chen, Haozhe Zhang, Sixian Li, Jingjing Gong, Zhenyang Lin, Ziyang Chen, Xipeng Qiu

**Published:** 2026-10-20

**Categories:** cs.RO

**Abstract:**

Can embodied task agents use tactile experience to perform fine manipulation without task-specific model training? We investigate this capability through Tactile-ETA, a system that incorporates tactile experience into an embodied task agent through in-context learning. Successful expert demonstrations show how contact changes during manipulation. We pair each recorded robot movement with its camera views and bilateral tactile observations. We keep tactile observations at the start and end of each movement and add two neighboring time points when the tactile images change markedly in between. The agent combines these demonstrations with current vision, tactile observations, and proprioception to select actions through general-purpose robot tools, without updating model parameters. Across four simulated manipulation tasks, adding tactile observations to the same two expert demonstrations increases average success from 15.75% to 24.00%, with current sensing and execution interfaces held fixed. Further experiments show that performance depends on the sensory content and representation of demonstrations, while comparing agent models on Lift Bottle reveals substantial differences under the same system configuration. On the real-robot tasks, demonstrations with aligned visual and tactile observations increase the task success rate from 0% to 60%. These findings demonstrate the feasibility of using tactile experience in context for fine manipulation and suggest a path toward extending embodied task agents to additional sensory inputs.

### 论文解读

#### 摘要翻译
本文研究具身任务智能体能否利用触觉经验完成精细操作，而不针对任务训练模型。作者提出Tactile-ETA，把成功专家演示中的视觉、动作和双指触觉按动作片段对齐，再作为上下文提供给冻结的智能体。四项仿真任务中，加入演示触觉提升平均成功率；真实机器人擦拭和拾取任务也展示了初步可行性。

#### 方法动机分析
视觉和动作轨迹往往没有说明接触发生了什么，智能体因此难以判断何时调整姿态或结束一个操作阶段。作者的核心假设是，若把触觉变化与对应的动作、视觉和状态同步，智能体就能从示范中学到接触过程的线索，并在当前操作中复用，而无需更新模型参数。

#### 方法设计详解
输入是成功专家回合的头部/腕部相机图像、机器人状态、测得的末端运动和左右指尖触觉序列。系统先按动作边界切分回合，保留每段视觉与触觉的首尾，并根据触觉图像变化筛选少量中间时刻。具体做法是在触觉图像区域内跟踪3×4网格图像块，以位移第75百分位表示变化；跟踪质量低时改用灰度差。超过阈值才加入变化点前后的相邻样本，因此每个指尖每段最多提供四帧。构造好的演示只在回合开始时提供一次。运行中，冻结的智能体结合这些演示、当前视觉/触觉、本体状态和动作执行反馈，通过通用机器人工具选择下一动作；闭环反馈继续进入后续决策。

#### 方法对比分析
其创新贡献是把有时间顺序的接触变化与视觉和运动记录作为一个演示单元。区别于只提供视觉和运动记录的做法，Tactile-ETA显式纳入触觉；区别于无差别保留所有帧的做法，它用事件筛选控制上下文长度。该方法更适合依赖接触状态判断的插入、抓取和旋转任务，但效果依赖传感器图像质量、演示对齐和基础智能体能力。

#### 实验分析（精简版）
作者在四项UniVTAC任务上以每项100次评估比较相同两段专家演示：视觉-动作演示平均成功率15.75%，加入筛选触觉后为24.00%。举瓶任务中，筛选表示为36%，仅端点为27%，保留全部触觉帧为2%，显示输入组织方式很关键。真实机器人两项任务各每种条件5次，有演示时均成功3次、无演示时均为0次；真实试验规模较小，证据仅说明初步可行性。

#### 实用指南
论文关注无需微调的上下文方案。复现时需保留专家演示的动作边界、相机视角、状态坐标和触觉时间戳，并校准图像ROI、位移阈值及低质量图像的差分回退。主实验使用GPT-6低推理强度、两段成功演示和通用动作工具。迁移到不同触觉传感器或机器人时，应重新检查图像尺度、采样节奏和任务成功标准；演示数量增加不保证更好，需做表示与任务分层消融。

#### 总结
核心思想：把对齐的触觉经验放入操作上下文。

1. 将专家回合切成有边界的动作片段。
2. 按触觉变化挑出少量关键时刻并与视觉、动作对齐。
3. 把演示交给冻结智能体，结合当前感知闭环操作。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27613v1)
- [arXiv](https://arxiv.org/abs/2610.27613v1)

---

<a id='2610.27271v1'></a>
## [XoNav: An Exoskeleton as a Personalized Haptic Interface for Assisted Navigation](https://arxiv.org/abs/2610.27271v1)

**Authors:** Debadrata Sarkar, Soumen Sen, Aman Arora

**Published:** 2026-10-19

**Categories:** cs.RO

**Abstract:**

Walking without vision requires timely, unambiguous guidance, yet many wearable aids encode directions as audio or vibrotactile signals that users must translate into movement. We present XoNav, a four-actuator hip exoskeleton that renders guidance cues directly at the hip joints. It renders an Advance cue and two-level Braking cues through sagittal-plane actuators and two-level Steering cues through frontal-plane actuators. A bilateral LSTM-based gait-state classifier uses two thigh-mounted IMUs to identify gait states, enabling haptic cues to be delivered in coordination with the wearer's gait while suppressing them during standing. The classifier generalized to unseen users with recalls of 95.2-97.9%, with mean detection delays of 38.6 and 63.1 ms for stance-to-swing and swing-to-stance transitions, respectively. Constrained human-in-the-loop Bayesian optimization personalized Braking for minimum resistance and Steering for maximum comfort while ensuring detectability and safety. All participants reliably distinguished between grades of their personalized cues and the firmest Braking cue remained below 53% of the safety limit. In blindfolded Wizard-of-Oz navigation trials along three paths, the participants kept 98.8% of the walked distance within a 0.75 m wide passage, with a 4.9% increase in mean progression speed per trial. A body-worn camera, RTAB-Map and Nav2 further enabled a feasibility study of autonomous navigation and detours around previously unmapped obstacles.

### 论文解读

#### 摘要翻译
XoNav是一套髋部外骨骼，通过髋屈伸和外展/内收直接传递前进、减速和转向提示，并根据步态相位安排提示时间。系统以两枚大腿IMU识别步态，再用人在回路优化个体提示强度。研究包含盲行人工导航和小规模自主导航可行性测试。

#### 方法动机分析
音频或振动提示要求用户把符号转换为身体动作，音频还可能遮盖环境声。作者希望让髋部设备通过力觉直接表达方向与减速，同时跟随穿戴者发起的步态而不强行驱动。核心假设是，按步态相位呈现并为每个人调整提示，可以兼顾可辨认度、安全和舒适。

#### 方法设计详解
设备总重4.3千克，每侧髋部有屈伸与外展/内收执行器。屈伸力矩表达前进或两级刹车，外展动作表达左右转向；支撑/摆动和站立状态决定提示何时释放或抑制。分类器只用两枚大腿IMU，以100Hz采样，提取过去300毫秒信号的统计特征并输入共享LSTM序列分类器；左右腿各有分类头，连续两次预测一致才切换状态。人在回路的贝叶斯优化为制动和转向选择参数，制动设置需满足至少13/15次正确辨认，并先通过刚度安全上限筛查；转向还通过参与者成对舒适度比较选取。自主模式将胸前RealSense、RTAB-Map定位和Nav2路径转为离散髋部提示。

#### 方法对比分析
创新在于用髋部力矩传递方向和减速，而非仅给振动或音频信号，并让提示与步态事件同步。与固定提示相比，受约束的贝叶斯优化针对个体辨认能力和舒适度调节强度。它更适合作为穿戴者主动行走时的辅助提示，系统仍依赖定位和规划稳定性。

#### 实验分析（精简版）
两枚大腿IMU的步态分类器在十名受试者留一验证中的三类召回率为95.2–97.9%。十名健康男性完成150次蒙眼人工导航试验，98.8%的路程保持在0.75米通道内，模型估计每次重复速度提高4.9%。自主导航只有三位参与者、24次测试，15次无障碍中14次到达，9次障碍绕行中8次成功；出现定位漂移和一次延迟检测碰撞。

#### 实用指南
复现需对齐髋关节轴、步态事件和提示时序，按个体做可辨认度及舒适度评估，并在试穿时通过独立安全节点限制位置、增益和力矩。自主导航配置使用胸前相机、RTAB-Map和ROS 2 Nav2。当前试验在室内、蒙眼健康人中开展；应用于视障者前需要与白手杖对照并扩大环境和安全验证。

#### 总结
核心思想：用个体化髋部力觉引导行走。

1. 以大腿IMU识别步态状态。
2. 在合适相位输出前进、制动或转向力觉。
3. 按辨认、安全和舒适反馈个体化提示。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.27271v1)
- [arXiv](https://arxiv.org/abs/2610.27271v1)

---

