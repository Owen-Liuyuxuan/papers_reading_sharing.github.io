time: 20260923

# Arxiv Computer Vision Papers - 2026-09-23

## Table of Contents

1. [MachEmbodied-U0: Unified Understanding and Generation Model for Embodied Intelligence](#2609.25627v1)
2. [From Instrument-Mounted Demonstrations to In-Vivo Execution: Learning Bimanual Laparoscopic Appendectomy Without Robot-Collected Demonstrations](#2609.25625v1)
3. [Leveraging Vision-Based Point Cloud Map Priors for Camera-Based 3D Object Detection and Online Vectorized HD Mapping](#2609.26325v1)
4. [MatchFusion: Explicit-Implicit Instance Matching for Spatio-Temporal Multimodal Autonomous Driving](#2609.25860v1)
5. [Predict Before You Step: Auditable Occupancy Forecasting for Dynamic Obstacle Avoidance under Sparse Guidance](#2609.25969v1)
6. [Unsigned Distance Maps on 2D Point Cloud Registration](#2609.25932v1)
7. [Dual Covariance Gaussian Splatting SLAM: Decoupling Rendering and Registration for Robust Real-Time Tracking](#2609.25746v1)
8. [Hierarchical Floorplan-Guided Vision-Language Exploration for Embodied Question Answering](#2609.26360v1)
9. [Minimal Recurrent Behavioral Memory for Imitation under Partial Observability](#2609.25757v1)

---

## Papers

<a id='2609.25627v1'></a>
## [MachEmbodied-U0: Unified Understanding and Generation Model for Embodied Intelligence](https://arxiv.org/abs/2609.25627v1)

**Authors:** Haoran Wen, Wenfu Wang, Kunsong Shi, Jingke Wang, Wancheng Feng, Yiren Zhang, Yueran Zhao, Xuancheng Zhang, Nanfei Ye, Xingru Chen, Zhaohong Sun, Chengmin Yang, Zikang Yu, Penghao Bi, Jia Shi, Yu Liu, Kun Zhan, Yan Xie

**Published:** 2026-09-22

**Categories:** cs.RO, cs.CV

**Abstract:**

General-purpose robot control requires models to understand task intent, identify where to interact, capture how the scene evolves, and generate precise actions. Vision-language-action models provide strong semantic priors but typically do not explicitly model scene dynamics, while world-action models couple visual prediction with control without necessarily exposing the task-relevant semantic and spatial structure needed for fine-grained manipulation. We present MachEmbodied-U0 (ME-U0), a unified embodied foundation model connecting understanding and generation experts through a Mixture-of-Transformers architecture. Subtask prediction and affordance grounding guide joint visual-dynamics and action generation via flow matching. Visual dynamics encompass future RGB, depth, surface normals, and optical flow, providing complementary supervision for appearance, geometry, and motion. Multi-rate Rotary Position Encoding (MRPE) aligns visual dynamics with fine-grained control. We pretrain ME-U0 on approximately 4,200 hours of curated demonstrations from robotic datasets and egocentric datasets. Using only the supervision natively available in each downstream benchmark, ME-U0 achieves an average score of 17.66 on the RoboDojo simulation benchmark and average success rates of 99.0\% and 82.5\% on LIBERO and LIBERO-Plus, respectively. We additionally validate ME-U0 on real-world robotic manipulation tasks, demonstrating its effectiveness beyond simulation. Without corresponding downstream supervision, ME-U0 further demonstrates zero-shot subtask prediction, affordance grounding, and visual dynamics on simulated and real-world observations. Overall, ME-U0 combines competitive downstream control performance with transferable task-grounding and visual-dynamics capabilities across simulation and the real world.

### 论文解读
#### 摘要翻译
MachEmbodied-U0（ME-U0）是面向具身智能的统一理解与生成模型。它用混合 Transformer 连接理解专家和生成专家，在约4,200小时精选机器人与第一视角数据上预训练，统一处理子任务、交互目标、未来视觉动力学和动作。模型在 RoboDojo 得分17.66，在 LIBERO 与 LIBERO-Plus 的平均成功率分别为99.0%和82.5%，并展示零样本视觉预测与 affordance 接地。
#### 方法动机分析
VLA有语言语义却常不显式建模动力学，WAM能预测未来但缺少任务语义和空间接地；多机器人又有状态、动作维度和频率差异。论文的关键假设是：先明确“现在做什么、在哪里交互”，再让视觉未来与控制共同生成，能提升精细操作和长程执行。
#### 方法设计详解
给定指令、图像和机器人状态，视觉/语言编码器形成共享上下文。理解专家自回归生成当前子任务；有标注时同时预测目标框和 interaction point。生成专家接收这些语义空间条件，以条件 Flow Matching 联合去噪未来 RGB、深度、法线、光流潜变量及动作轨迹。MRPE 为压缩视觉帧和高频动作提供多速率时间位置，使视觉变化与动作子步对应。异构关节、夹爪、底座等变量先映射到统一接口，并以 mask 标记缺失维度。总目标由生成流匹配损失和理解交叉熵组成；还可训练正向/逆动力学。LIBERO 使用24步末端位姿控制，RoboDojo使用48步增量关节控制。
#### 方法对比分析
ME-U0区别于只输出动作的VLA，也区别于只把未来视频作为附加预测的WAM：它把子任务和 affordance 直接作为生成条件，并在同一骨干中联合几何、运动与动作。MRPE解决视觉时间压缩与控制频率不匹配，统一接口则支持跨 embodiment 预训练。代价是训练数据、标注和生成计算更复杂，对历史记忆仍不充分。
#### 实验分析（精简版）
RoboDojo-Sim上ME-U0总过程分17.66，高于WAM基线OpenWAM-α的17.18；精度和长程过程分为23.95、36.98，但低于DM0.5的总分24.90。标准LIBERO平均99.0%，LIBERO-Plus在无额外适配下为82.5%，相机扰动仅70.2%。RoboDojo记忆任务过程分8.42，显示历史信息缺失是主要短板。真实机器人结果为已见平台上的定性展示。
#### 实用指南
复现要统一机器人状态/动作语义，准备子任务、目标接地及几何运动标注，按任务桶采样，并正确实现MRPE和Flow Matching。论文提供项目页与GitHub链接，但未明确完整权重、依赖版本和复现实验命令。迁移到新机器人需替换 embodiment 映射、控制接口并重新后训练；未见平台泛化和推理延迟仍需单独验证。
#### 总结
核心思想：语义接地联合生成
速记：
1. 指令观测预测子任务与交互点；
2. 异构状态映射统一接口；
3. MRPE对齐视觉与动作时间；
4. 联合生成几何、运动和控制；
5. 用下游原生监督适配任务，并检查新平台的控制延迟。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.25627v1)
- [arXiv](https://arxiv.org/abs/2609.25627v1)

---

<a id='2609.25625v1'></a>
## [From Instrument-Mounted Demonstrations to In-Vivo Execution: Learning Bimanual Laparoscopic Appendectomy Without Robot-Collected Demonstrations](https://arxiv.org/abs/2609.25625v1)

**Authors:** Dongho Yee, Juahn Oh, Jinseok Lee, Jiyul Lee, Yechan Seo, Seong Jeong, Minsung Kim, Seonho Shim, Younghoon Noh, Hyuk Choi, Youngbin Kong, Kyu Eun Lee, Hyoun-Joong Kong

**Published:** 2026-09-22

**Categories:** cs.RO

**Abstract:**

Most minimally invasive surgery is still performed with hand-held laparoscopic instruments, and the surgeon's instrument kinematics are lost when the operation ends; only the endoscope video is kept. This paper presents an end-to-end pipeline that captures this motion in the operating room and uses it to train a surgical robot policy, validated on live animals. We introduce a surgical instrument-state logger that mounts on the shaft of a standard laparoscopic instrument and recovers its pose and jaw state from an inertial sensor, a time-of-flight sensor and a Hall sensor, with no external camera or tracker. A data pipeline measures the latency of every sensor channel against a robot ground truth and aligns the channels before forming observation-action pairs. On these demonstrations we train a diffusion policy with a fine-tuned DINOv3 backbone, selecting its design by closed-loop rollouts in a physics simulator reconstructed from depth maps of an ex-vivo rabbit appendix. The policy is then retrained on 849 in-vivo demonstrations from four live rabbits and deployed on four additional live rabbits with electrosurgery armed. With the surgeon selecting the surgical phase, the policy completed the appendectomy in three of the four animals. The results show that demonstrations recorded from a surgeon's own instruments are sufficient to train, select and deploy a bimanual surgical policy in vivo. The robot serves only as the timing reference for sensor calibration and as the executor, and collects no demonstrations. Both demonstration corpora are released to support future surgical robot learning research.

### 论文解读
#### 摘要翻译
论文提出一种不依赖机器人遥操作演示的双臂腹腔镜学习方案：在普通手动器械上安装状态记录器，结合内窥镜视频训练策略，并迁移到真实机器人，完成兔子活体阑尾切除。

#### 方法动机分析
手动腹腔镜手术广泛存在，但器械运动学通常不会被保存；用手术机器人采集演示又昂贵。作者利用套管提供的远程运动中心约束，假设只要可靠记录器械的姿态、深度和钳口状态，就能把医生动作转为机器人可执行的示范，同时以阶段条件处理复杂长流程。

#### 方法设计详解
记录器由 IMU、ToF 和 Hall 传感器组成，输出滚转、俯仰、插入深度和钳口开合。系统先测量各通道延迟，用一阶滞后加纯延迟模型补偿，再把传感器和 30 fps 视频对齐，重采样到 10 Hz。策略输入 320×192 内窥镜图像、双臂状态 [cos(rho), sin(rho), theta, d/dmax] 及手术阶段；输出未来 16 步（约 1.6 秒）的端点增量、钳口开合和电切开关。1D U-Net 扩散策略生成动作序列，FiLM 注入阶段信息，视觉骨干使用 DINOv3 ViT-B/16，并主要冻结其参数。扩散采样的意义是保留同一视觉场景下多种合理操作轨迹，而不是回归一个平均动作；动作块则减少逐帧预测的抖动。推理闭环为 5 Hz，中位延迟 23.5 ms。

#### 方法对比分析
与依赖机器人采集示范的模仿学习相比，该方法把数据采集移到廉价、可穿戴的手动器械上；与只用视频的方法相比，它补充了深度、姿态和夹持状态。DINOv3 冻结式迁移和闭环仿真筛选也是关键设计，说明低离线误差不等于真实安全。它适合有明确阶段、RCM 约束和器械状态可测的机器人手术，但跨术者、器械和解剖结构的泛化仍待验证。

#### 实验分析（精简版）
数据包括 2 只兔尸体的 533 个 episode（74 分钟）和 4 只活兔的 849 个 episode（135 分钟）。Isaac Sim 筛选中，DINOv3 仅解冻最后一块在全流程接近任务成功率为 81%，ResNet-18 为 12%。活体部署 4 例完成 3 例，RCM 峰值误差 0.84 mm，低于 5 mm 安全阈值；失败案例与逆运动学分支切换触发限位保护有关。样本很小，且阶段预测存在领域偏移。

#### 实用指南
论文提供 Rosota-Research/ICRA2027-Invivo-main 仓库，包含 Ex-vivo 和 In-vivo 演示语料库。复现时需实现传感器标定、延迟匹配、视频同步、RCM/关节限位安全约束和能量互锁；论文未完整说明所有优化器、批大小和学习率。迁移到其他器械或动物时，应重新标定几何与延迟，重采集阶段分布，并重新训练或微调视觉策略。

#### 总结
核心思想：手动器械示范驱动活体机器人手术
1. 记录器械姿态、深度与夹持状态。
2. 延迟校准并与视频、手术阶段对齐。
3. 用阶段条件扩散策略预测动作块。
4. 以闭环仿真筛选视觉骨干，再执行活体手术。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.25625v1)
- [arXiv](https://arxiv.org/abs/2609.25625v1)

---

<a id='2609.26325v1'></a>
## [Leveraging Vision-Based Point Cloud Map Priors for Camera-Based 3D Object Detection and Online Vectorized HD Mapping](https://arxiv.org/abs/2609.26325v1)

**Authors:** Markus Käppeler, Rohit Mohan, Abhinav Valada

**Published:** 2026-09-22

**Categories:** cs.CV, cs.RO

**Abstract:**

Camera-based 3D object detection and online vectorized HD mapping provide compact scene representations for autonomous driving, but both depend on accurate metric geometry and remain limited by depth ambiguity. Over long-term deployment, observations from repeated traversals can be accumulated into persistent point cloud priors that provide geometric context beyond the current observations. Existing explicit point cloud prior approaches, however, rely on LiDAR-based map construction and therefore require expensive 3D ranging sensors. We propose a framework that constructs a static point cloud prior map from previous camera traversals using Pi3X and augments each point with DINOv3 features. At runtime, a local prior patch is retrieved using global localization, encoded with a sparse voxel backbone, and fused in bird's-eye view (BEV) with lifted multi-view camera features. Task-specific sparse transformer heads then predict 3D objects and vectorized map elements from the fused representation. On Argoverse 2, the vision-based prior improves a strong baseline from 0.287 to 0.299 CDS and from 0.669 to 0.750 vectorized mapping mAP. Ablations show that semantic DINOv3 features are particularly important for vectorized mapping. These results demonstrate that vision-built geometric-semantic priors provide an effective form of long-term scene memory for camera-based perception, improving both tasks without LiDAR for prior-map construction or online inference.

### 论文解读
#### 摘要翻译
本文提出用历史摄像头观测重建视觉点云地图，把车辆重复行驶形成的长期信息作为先验，增强相机 3D 目标检测和在线矢量化高精地图生成。Pi3X 负责几何重建，DINOv3 提供语义特征。在 Argoverse 2 上，检测 CDS 从 0.287 提升至 0.299，矢量建图 mAP 从 0.669 提升至 0.750。
#### 方法动机分析
单帧视觉存在深度歧义，车辆或遮挡物还会挡住车道线等静态结构；传统点云先验又常依赖昂贵 LiDAR。作者的核心假设是：车辆有可靠全局轨迹，且道路静态结构在多次经过间基本不变，因此历史观测能提供当前画面缺失的几何和语义上下文。
#### 方法设计详解
离线时，Pi3X 从历史视频预测局部 3D 点图和置信度，用轨迹真值估计尺度因子恢复度量尺度；DINOv3 ViT-S 将语义附加到点上，并用 PCA 把 384 维压到 64 维。动态点过滤后，静态点以 0.2 m 体素池化，形成平铺全局地图。在线时根据当前位姿检索局部点云并变换到车体坐标。图像经 VoVNet-99 和 LSS 变成 BEV，地图点（位置、置信度、RGB、DINO 特征）经稀疏 3D 骨干编码为地图 BEV；两者拼接后用 Conv-BN-ReLU 融合，再由 SparseDrive 风格稀疏 Transformer 同时输出 3D 框和矢量多段线。训练采用 AdamW、学习率 2e-4、80 epochs、batch size 24。
#### 方法对比分析
与无先验的纯视觉基线相比，本文引入可检索的长期视觉地图；与 LiDAR 地图先验相比，地图构建不需要激光雷达。消融表明，性能提升并非仅来自几何点的位置：DINOv3 语义对识别车道等静态元素尤其关键。方法适合重复行驶、地图结构相对稳定且位姿可用的场景。
#### 实验分析（精简版）
Argoverse 2 验证集、480×704 输入下，SparseDrive 的检测 CDS/建图 mAP 为 0.287/0.669，本文方法为 0.299/0.750；640×960 下检测 CDS 达 0.314，接近 LiDAR 先验方法 DualViewMapDet 的 0.311。仅几何先验时建图 mAP 为 0.683，加入 DINOv3 后达到 0.756，说明语义信息对地图元素分类和连贯的矢量输出起主要作用。覆盖率更高的区域中，CDS 提升最高约 1.9 个百分点，但结果仍依赖给定轨迹和静态场景；论文没有验证施工、季节变化或定位明显偏差时的性能。
#### 实用指南
复现需先用 Pi3X 重建历史视频、做尺度校准和动态点过滤，再提取 DINOv3 特征并构建体素化地图；推理时按位姿检索局部地图并联合训练融合模块。论文使用 VoVNet-99、SparseDrive 等组件，但未明确给出本文完整代码链接，不能断言项目已开源。迁移到新数据集时需重建地图、重新校准坐标和训练任务头，同时检查地图覆盖率、点云置信度与坐标系定义，否则先验可能引入错误几何。
#### 总结
核心思想：视觉地图补足单帧深度
1. 历史视频重建并尺度校准视觉点云。
2. 为静态点附加压缩的 DINOv3 语义。
3. 按当前位姿检索地图，编码成 BEV 先验。
4. 与图像 BEV 融合，联合检测目标和矢量地图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.26325v1)
- [arXiv](https://arxiv.org/abs/2609.26325v1)

---

<a id='2609.25860v1'></a>
## [MatchFusion: Explicit-Implicit Instance Matching for Spatio-Temporal Multimodal Autonomous Driving](https://arxiv.org/abs/2609.25860v1)

**Authors:** Xiaoyu Li, Jiajia Fu, Long Shi, Tianyu Du, Ruihang Li, Xian Wu, Lijun Zhao, Yingtao Zhang, Lining Sun, Ruifeng Li

**Published:** 2026-09-22

**Categories:** cs.CV, cs.RO

**Abstract:**

Sparse instance representations provide a compact interface for spatial LiDAR-camera and temporal past-current interaction in multimodal perception and E2EAD. Effective interaction requires reliable instance correspondences despite geometric discrepancies and heterogeneous semantic representations. Attention-based methods exploit contextual semantics but often require specialized representation alignment, increasing computational overhead. In contrast, association based on structured object states is efficient and interpretable but lacks contextual evidence to resolve ambiguous matches. To combine these complementary strengths, we propose MatchFusion, a learnable instance matching and fusion module for spatio-temporal multimodal autonomous driving. MatchFusion initializes pairwise affinities using geometric similarity and category consistency, then selectively refines structurally plausible associations using instance embeddings. The resulting soft matchmap guides a common residual aggregation operator for adaptive information exchange. This unified matching-fusion formulation supports spatial LiDAR-camera and temporal past-current interaction, using multi-view image-plane geometry and motion-compensated BEV geometry as the respective structural priors. Experiments on nuScenes demonstrate consistent perception gains across diverse front-end configurations. Compared with a prior instance-centric fusion method, the MatchFusion-equipped system achieves higher perception accuracy while reducing FLOPs by 55.3% and GPU memory usage by 39.3%, with the matching-fusion module accounting for only 3.7% of total perception latency. Integrating temporal MatchFusion into SparseDrive further improves perception within an E2E framework without additional supervision. These results establish explicit-implicit matching as an effective and efficient mechanism for spatio-temporal instance interaction.

### 论文解读
#### 摘要翻译
MatchFusion 面向自动驾驶中的 LiDAR—相机及跨时间信息融合，把几何结构关联与实例查询语义结合，学习可靠的实例对应关系，并用于 3D 检测、跟踪和端到端驾驶。

#### 方法动机分析
密集特征交互计算昂贵，单纯几何匹配又容易受深度误差、运动和标定误差影响。论文的关键假设是：几何和类别先验负责筛选候选，查询嵌入负责在候选内做语义修正，从而兼顾效率与匹配可靠性。

#### 方法设计详解
输入是两组实例，每个实例含 3D 框状态（中心、尺寸、偏航角、类别）和查询向量。首先计算几何相似度，并用类别一致性掩码屏蔽跨类配对；对剩余候选，将两侧查询嵌入映射后送入 MLP 得到语义修正，再经 sigmoid 形成软 matchmap。该图用于残差聚合：保留当前实例查询，同时按匹配权重加入另一组实例的变换特征。空间分支把 3D 框投影到图像平面，用相对偏移和尺寸比构造几何编码，以减轻深度误差；时间分支通过自车运动和目标运动补偿，在 BEV 中对齐历史与当前实例。模块可单独做空间（S）、时间（T）或联合时空（ST）融合。实验配置为 16 epoch、batch size 32、AdamW 初始学习率 1e-3、余弦退火，使用 VoxelNet 和 ResNet-50。其关键在于先缩小匹配搜索空间，再让语义模块处理不确定候选，避免无约束的全局交互。

#### 方法对比分析
相比 TransFusion、DeepInteraction 等稠密交互，MatchFusion只在实例之间交换信息；相比 SparseFusion，它把显式几何/类别约束和隐式查询语义修正串联起来，并采用软匹配引导的残差融合。方法适合已有实例检测结果、坐标变换可靠的多传感器或多帧系统，但对旋转标定误差仍较敏感。

#### 实验分析（精简版）
nuScenes 验证集上，配合 Sparse4Dv3 相机前端时，MatchFusion-ST 达到 73.0 NDS、70.4 mAP，较对应基线提升 2.9 和 5.3。与 SparseFusion 相比，FLOPs 从 569.1G 降到 254.2G，显存从 6.1GB 降到 3.7GB。去掉类别掩码后 NDS 从 73.0 降为 72.1；残差融合（72.5）也优于简单拼接（69.4）。端到端 SparseDrive 中 AMOTA 提升 2.7，但大旋转误差会削弱收益。结果同时支持精度和效率两项主张，但这些结论主要来自 nuScenes，尚不足以证明跨城市、跨传感器标定条件下的泛化。

#### 实用指南
论文仅声明代码将发布，未给出可核验仓库链接。复现需准备 nuScenes 的多视角图像、LiDAR、实例框、类别及运动补偿信息，并严格处理相机—LiDAR标定。迁移时要替换坐标投影、BEV 对齐和类别映射，并重训语义修正模块；旋转标定应重点测试。完整依赖、随机种子和增强细节未说明。

#### 总结
核心思想：显式几何筛选，隐式语义修正

1. 编码带框状态与查询的跨模态、跨时间实例。
2. 以几何相似度和类别一致性筛选候选。
3. 用 MLP 语义细化并经 matchmap 做残差传递，完成空间/时间融合。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.25860v1)
- [arXiv](https://arxiv.org/abs/2609.25860v1)

---

<a id='2609.25969v1'></a>
## [Predict Before You Step: Auditable Occupancy Forecasting for Dynamic Obstacle Avoidance under Sparse Guidance](https://arxiv.org/abs/2609.25969v1)

**Authors:** Yuhui Mao, Fen Liu, Shenghai Yuan, Tianxin Hu, Ruimeng Liu, Rong Su

**Published:** 2026-09-22

**Categories:** cs.RO

**Abstract:**

Legged robots under sparse waypoint guidance must avoid moving obstacles using partial, rapidly changing LiDAR observations. We present LOOP (Latent-recurrent Occupancy rollOut Policy), a local avoidance policy that connects sparse waypoint guidance to a frozen locomotion controller at 50 Hz. From occupancy and ego-velocity histories, a recurrent predictor forecasts future occupancy over a 1 s horizon by warping the current map with learned flow and visibility gates. These maps guide velocity selection through map-derived features and geometric risk estimates, providing an explicit interface for inspecting and replacing predictions. In encounter-synchronised Isaac Lab evaluations, LOOP achieves 57.1% head-on success at obstacle speeds of 2.5-3.2 m/s, exceeding a retrained reactive baseline by 8.2 percentage points. Comparisons with a rollout-free BEV policy show smaller, scenario-dependent gains from the prediction branch, including improved crossing success and reduced variability across training seeds at the highest head-on speeds. The adapter runs onboard a Unitree Go2 in 14.5 ms per step and completes all 16 real-world crossing trials without collision, demonstrating deployment feasibility.

### 论文解读
#### 摘要翻译
论文提出 LOOP（Latent-recurrent Occupancy rollOut Policy），让四足机器人在稀疏路点引导下，仅凭局部 LiDAR 观测预测未来 1 秒的占据变化，并在 50 Hz 控制循环中规避动态障碍。它在正面高速场景（障碍物 2.5–3.2 m/s）取得 57.1% 成功率，比重新训练的反应式基线高 8.2 个百分点；部署在 Unitree Go2 上每步耗时 14.5 ms，16 次真实横穿试验均无碰撞。

#### 方法动机分析
驱动力是 VLM/VLA 或规划器更新不超过 1 Hz，无法覆盖路点更新之间的突发冲突；现有反应式策略又难以解释提前减速的依据。LiDAR 只提供局部、带遮挡的几何表面，不直接给出障碍物速度。LOOP 的核心假设是短时历史足以外推局部运动，关键思路则是把“未来会出现什么”显式变成可检查的地图，并让它成为预测分支影响决策的唯一通道。

#### 方法设计详解
输入为机器人中心 ±4 m、32×32 的 BEV 占据图和本体速度历史：12 帧、间隔 0.2 s，另加路点偏移。历史 LSTM 提取状态后，预测 LSTM 自展开 5 步；每步预测流场和可见性门控，用 warp 重采样当前地图，得到 0.2–1.0 s 的未来占据图。预测图一方面编码成未来 token，另一方面在 3×3 候选速度的机器人圆盘轨迹上做平均占据风险，形成 45 维风险向量，连同路点特征输入决策 MLP，输出离散速度模式，再由冻结的运动控制器执行。预测目标采用同一 LiDAR 渲染流程生成，并用占据加权平方误差训练（β=3）；策略用 PPO 训练，预测与决策在 50 Hz 推理。完整适配器 0.94 M 参数，Jetson AGX Orin 前向延迟 14.5 ms。

#### 方法对比分析
相比 REASAN 等反应式射线策略，LOOP 监督显式未来占据；相比 ABS 屏蔽，它把几何风险直接融入速度选择；相比大型占据世界模型，它只在局部 50 Hz 环路中做轻量 1 秒预测。地图接口还允许推理时替换为全零或真值，检查预测通道对动作的影响。

#### 实验分析（精简版）
在相遇同步的 Isaac Lab 评测中，LOOP 在正面 std/fast/vfast 三档成功率为 59.9/57.2/57.1%，重新训练 REASAN 为 54.7/48.6/48.9%；最高速提升 8.2 个百分点，跨种子标准差也由 4.9 降至 1.6。去掉 rollout 后三类场景成功率分别下降 1.5、3.1、1.2 个百分点。将预测地图置零会平均损失 31.1 个百分点，显示策略确实依赖该通道；但 0.8 m/s 低速时 LOOP 只有 47.7%，反而低于无预测版本的 57.3%，说明预测器受训练速度分布限制。实机结果只有 16 次试验，主要证明可部署性。

#### 实用指南
论文给出了 Isaac Lab、传感器栅格化、网络规模、PPO 训练和 Orin 延迟设置，但未明确提供代码或权重开源链接。复现时要保留 12 帧历史、0.2 s 步长、5 步 horizon、0.9–1.9 m/s 训练速度，并采用相遇同步协议。迁移机器人需重做 LiDAR-BEV 标定、机器人包络风险探针、速度模式和运动控制器接口。

#### 总结
核心思想：用可审计未来地图驱动避障

速记 pipeline：
1. 编码历史 LiDAR 与速度。
2. 用 LSTM 流场滚动预测未来地图。
3. 融合地图 token 和几何风险选速度，再用零地图/真值审计贡献。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.25969v1)
- [arXiv](https://arxiv.org/abs/2609.25969v1)

---

<a id='2609.25932v1'></a>
## [Unsigned Distance Maps on 2D Point Cloud Registration](https://arxiv.org/abs/2609.25932v1)

**Authors:** Ricardo B. Sousa, Giorgio Grisetti, Héber Miguel Sobreira, Carlos André Silva, António Paulo Moreira

**Published:** 2026-09-22

**Categories:** cs.RO

**Abstract:**

2D point cloud registration arises in laser odometry and Simultaneous Localization and Mapping (SLAM) for mobile robots. Iterative Closest Point (ICP) is one of the most widely used approaches. Still, its iterative procedure recomputes correspondences via nearest-neighbor search at every iteration, whereas correspondence-free alternatives focus on scan-to-map alignment. This paper proposes a 2D point cloud registration approach based on unsigned distance maps, precomputing the Euclidean distance to the nearest reference point, along with its spatial derivatives, over a discrete grid, replacing the per-iteration search with O(1) lookups. Moreover, point-to-point and point-to-plane error formulations are derived on the SE(2) manifold and solved via Gauss-Newton optimization. On a synthetic benchmark and the real-world IILABS 3D dataset, the precomputed point-to-point variant outperforms its analytical counterparts, achieving competitive laser-odometry drift compared to point-to-plane formulations, as the precomputed gradient regularizes correspondences in the presence of sensor noise.

### 论文解读
#### 摘要翻译
二维点云配准服务于激光里程计和SLAM。传统ICP每轮优化都要重新做最近邻搜索；无对应关系方法又常局限于scan-to-map。本文预计算离散网格上的无符号距离及空间导数，以O(1)查表替代搜索，并在SE(2)流形上统一推导点对点、点对平面残差，用Gauss–Newton求解。实验显示，预计算点对点方法在噪声下更有竞争力。

#### 方法动机分析
ICP的在线对应关系计算既耗时，也会受扫描采样差异影响。作者的核心假设是参考点云可先转换为空间距离场，使“寻找对应点”隐含在查询中，同时保留标准最小二乘优化接口。代价是网格量化和有限差分偏差，且距离场在等距点处不可微。

#### 方法设计详解
给定参考点云作为输入，先用8邻域最佳优先波前传播，从占据格子向外写入最近点索引，可限制在截断距离内；再用中心差分得到梯度和Hessian。M0只存索引、查询时解析计算；M1存距离和梯度；M2进一步存Hessian。移动点经当前SE(2)位姿投影后，直接查询距离场，形成“投影—查表—组装残差与Jacobian—流形更新”的模块化流程，输出更新后的位姿。点对点残差就是查询距离；点对平面残差将旋转后的移动点法向与“归一化梯度×距离”相乘。扰动采用右乘更新，在srrg2_solver中进行Gauss–Newton。合成实验最多迭代50次、步长阈值10^-5、截断距离5 m；真实实验使用0.03 m网格和0.50 m截断距离。

#### 方法对比分析
M0本质上是用O(1)网格索引替代ICP的kd-tree关联；M1的有限差分梯度提供平滑正则化，使点对点方法在噪声下更稳；M2以存储换取二阶导数。M0点对平面与经典点对平面ICP在理论上等价，适合重视精度和不确定性标定的场景；M1适合希望减少显式法向依赖、兼顾鲁棒性的激光里程计。双线性插值可能受零距离格子偏置，不能默认开启。

#### 实验分析（精简版）
合成实验在circle、square、corridor三种场景进行100次试验，噪声为0或0.03 m。点对平面总体优于点对点；在真实IILABS 3D数据上，M1点对点在Nav A Diff、50%内点阈值下RTE/RRE为0.95%/0.078°/m，解析点对点ICP为1.32%/0.116°/m。其最好结果在Nav A Omni为0.81%/0.063°/m、Slippage为0.55%/0.056°/m；但Loop门口等不连续区域仍逊于点对平面方法。论文尚未给出与kd-tree的最终运行时延对比。

#### 实用指南
论文给出srrg2_solver实现框架和ricoslam链接，但仓库标注为“to be released”，开源状态需谨慎确认。复现时应固定0.03 m网格、截断距离、体素化和关闭双线性插值，并复现0.15 m法向邻域、鲁棒核及关键帧策略。迁移到其他二维激光器需重新调量程、噪声和网格；该方法不能直接替代三维或动态环境模型。

#### 总结
**距离场替代在线对应**
1. 波前传播建立无符号距离图。
2. 预存距离及有限差分导数。
3. 投影点用O(1)查询构造残差。
4. SE(2)流形优化并以漂移验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.25932v1)
- [arXiv](https://arxiv.org/abs/2609.25932v1)

---

<a id='2609.25746v1'></a>
## [Dual Covariance Gaussian Splatting SLAM: Decoupling Rendering and Registration for Robust Real-Time Tracking](https://arxiv.org/abs/2609.25746v1)

**Authors:** Edward Beng Wai Tan, Siew-Kei Lam

**Published:** 2026-09-22

**Categories:** cs.RO, cs.CV

**Abstract:**

ICP-based 3D Gaussian Splatting (3DGS) SLAM tracks in real time by registering incoming frames against map Gaussians, using each primitive's covariance for both rendering and registration. These two uses place conflicting demands on one covariance. The mapper shapes it to minimize photometric error, often flattening it against surfaces, while robust registration typically benefits from measurement uncertainty. We propose a dual-covariance parameterization. Each Gaussian keeps a single mean but holds two covariances: a rendering covariance optimized by the mapper, and a tracking covariance derived from an RGB-D sensor noise model. We further use the tracking covariances as Gaussian anchors for image corners, providing constraints in directions where depth geometry is weak. We evaluate on TUM RGB-D, ScanNet, Replica, and two outdoor sequences recorded with a RealSense D435i on wheeled and handheld platforms. We achieve robust tracking performance across multiple scenes and reduced odometry drift, while tracking at $\sim$ 60 FPS.

### 论文解读
#### 摘要翻译
论文提出 Dual Covariance Gaussian Splatting SLAM，解决 ICP 型 3DGS-SLAM 同一协方差兼顾渲染与配准的冲突：映射为降低光度误差会压扁高斯，而配准需要真实传感器不确定性。每个高斯共享均值、分别保存渲染协方差和 RGB-D 噪声驱动的跟踪协方差，并以高斯为视觉角点锚点，在 TUM RGB-D、ScanNet、Replica 及 RealSense 室外序列上实现稳健、约 60 FPS 的跟踪。

#### 方法动机分析
共享协方差会沿表面法向产生虚假高精度；单平面或无纹理场景又可能缺少位姿可观测方向，导致 ICP 退化、快速运动时跟踪失败。作者的驱动力是同时改善误差权重和几何可观测性，将“适合成像的形状”和“可信的测量误差”解耦，并用图像信息补充深度几何无法约束的方向。其适用边界是依赖 RGB-D 噪声模型的在线局部跟踪，而非保证大场景全局一致性。

#### 方法设计详解
输入是 RGB-D 帧、相机内参和已有高斯地图，输出是当前 SE(3) 位姿及更新后的渲染地图。对每个高斯维护渲染协方差 Σr 与跟踪协方差 Σt。后者由 RGB-D 噪声模型计算：射线距离为 z、焦距为 f、横向像素误差为 s 时，横向方差为 (zs/f)²，纵向方差由 σz(z)=A+Bz² 给出；最近观测锚点采用 p=0.5 px。当前深度点的观测协方差还加入最近邻半径在切平面内造成的歧义，地图匹配用 Σt 与观测协方差组成 G-ICP 权重。与此同时，KLT 跟踪图像角点，将角点反投影到由 Σr 最小轴确定的高斯切平面，匹配精度设为 2 px，形成带 Σt 的图像残差。深度残差与 Huber 鲁棒图像残差共同在 SE(3) 上用 Levenberg–Marquardt 求解；Σr 继续用于在线渲染和建图，KLT 地标丢失后移除。在 i9-13900KF+RTX 4090 上，平均处理速度约 59.1 FPS，论文未报告训练 epoch 或学习率，因为核心流程是在线估计而非离线训练。

#### 方法对比分析
相较 GS-ICP、SGAD-SLAM 等共享协方差方法，本方法不让映射器改变配准的传感器权重，核心创新是同一均值上的双协方差参数化；相较仅检测退化的 Spectral GS-SLAM，它同时修正权重并添加视觉锚点；相较 Photo-SLAM、MonoGS 等光度/特征跟踪方法，则保留 ICP 的实时速度与 3DGS 建图。Σt 改善不确定性建模，KLT 负责补充弱几何，两者不是同一机制：前者即使单独使用也能改善 Hessian 条件，后者在深度不可观测方向提供额外约束。

#### 实验分析（精简版）
在 i9-13900KF 和 RTX 4090 上，标准 TUM 平均 ATE 为 1.91 cm、速度 59.1 FPS；GS-ICP 为 2.4 cm、62.7 FPS。无纹理近场中，本文 ATE 1.67 cm，而 GS-ICP 达 194.94 cm。ScanNet 平均 ATE 为 9.61 cm；TUM 平均 PSNR/SSIM/LPIPS 为 21.39/0.784/0.219。消融显示共享协方差在无纹理序列为 194.96 cm，仅使用 Σt 为 192.38 cm，完整方法降到 1.67 cm，说明视觉锚点对深度不可观测场景关键。限制是缺少 BA，长序列一致性和远距离深度噪声下的室外精度有限。

#### 实用指南
基线使用公开实现和默认配置，但论文未说明本文代码或预训练模型已开源。复现需实现 RGB-D 内参、深度噪声律、0.5 px 观测锚点、2 px KLT 噪声及两套协方差；评价时注意 FPS 排除了输入预处理和后处理。换用其他深度相机应重新标定噪声，换成单目则需重构深度观测与不确定性模型。

#### 总结
核心思想：渲染跟踪协方差解耦
1. Σr 服务光度渲染。
2. Σt 表达 RGB-D 测量可信度。
3. Σt 加权 G-ICP 配准。
4. KLT 角点锚到高斯切平面。
5. 联合优化位姿并实时建图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.25746v1)
- [arXiv](https://arxiv.org/abs/2609.25746v1)

---

<a id='2609.26360v1'></a>
## [Hierarchical Floorplan-Guided Vision-Language Exploration for Embodied Question Answering](https://arxiv.org/abs/2609.26360v1)

**Authors:** Albert Gassol Puigjaner, Kostas Alexis

**Published:** 2026-09-22

**Categories:** cs.RO

**Abstract:**

Embodied Question Answering (EQA) requires an agent to explore a previously unseen environment, gather relevant information, and answer questions about the scene. Recent approaches leverage Vision-Language Models (VLMs) together with semantic maps or scene graphs to guide exploration. However, exploration is typically driven only by local observations, while structural priors about the environment remain largely unused. We propose HFLEX-EQA, a hierarchical EQA framework that combines online scene graph construction, VLM- based planning, semantic frontier exploration, and floorplan priors. The system incrementally builds a hierarchical scene graph and an open-vocabulary occupancy map from RGB-D observations, enabling a VLM to jointly reason over the scene graph, task-relevant visual observations, exploration history, and an estimated topological floorplan. Furthermore, we introduce a room-discovery strategy that leverages the floorplan and open-vocabulary frontier semantics to guide exploration toward semantically relevant yet currently unobserved room types. We evaluate HFLEX-EQA on the OpenEQA and ExploreEQA benchmarks and demonstrate deployment on a quadruped robot in real indoor environments. Our results demonstrate the benefit of combining VLM-based hierarchical planning with structural floorplan priors for the EQA task.

### 论文解读
#### 摘要翻译
具身问答要求机器人在陌生环境中主动探索并回答问题。HFLEX-EQA把在线场景图、VLM规划、语义前沿探索与拓扑楼层平面先验结合，使机器人能寻找尚未观测但语义相关的房间，并在 OpenEQA、ExploreEQA 及四足机器人实验中验证。

#### 方法动机分析
驱动力在于答案可能藏在已知房间、已见物体的盲区，或完全未发现的房间。只依赖局部图像容易盲目遍历或过早作答，现有方法也较少利用环境结构。作者的核心假设是：允许有误的楼层平面仍可作为弱先验，把场景图视为实时局部证据，由 VLM 判断当前缺失哪类信息。

#### 方法设计详解
RGB-D观测经Hydra风格流程增量构建分层场景图：房间保存最多3个与问题相关的CLIP视图，物体由YOLOe检测，前沿和占据图提供导航基础。高层VLM读取场景图、楼层图、视觉记忆、问题和历史，输出三种模式：explore_room在已知房间按目标物体语义给前沿排序；go_to_objects根据物体可见面积和图像覆盖率采样最佳检查视点（ε=1，权重0.7/0.3）；find_room计算候选房间到目标房间的拓扑最短距离，用exp(-距离)形成进度权重，再融合“通往目标房间”的视觉提示、房间类别相似度和未知区域大小选择前沿。答案置信度达到0.8才终止。

#### 方法对比分析
ExploreEQA和GraphEQA主要依据局部图像或场景图进行单一探索。HFLEX-EQA的关键不是简单把楼层图放进提示词，而是让高层决策选择专门低层策略，并把拓扑距离落实为房间发现前沿评分，因此能处理“目标房间尚未出现”的问题。代价是依赖感知、CLIP和VLM链路，错误楼层图或错误语义仍可能误导，尤其在长走廊和相似房间中。

#### 实验分析（精简版）
在HM3D上的108个OpenEQA和114个ExploreEQA episode、50次迭代预算中，Gemini-3.5选择题设置下HFLEX-EQA成功率分别为75.0%和63.2%；去除楼层先验后降为64.8%和53.5%，去除视点选择后为67.6%和58.8%。真实四足机器人覆盖计数、文字识别、定位和功能理解等问题；无选项重放时，50次规划有47次模式选择与有选项一致。局限是视觉退化会降低最终答案可靠性。

#### 实用指南
论文未说明代码或模型开源。复现需Habitat/HM3D、OpenEQA/ExploreEQA、YOLOe、LSeg、TSDF/占据图、层级场景图和VLM API；模拟楼层图只含房间标签与连通性，不含度量定位。关键超参数包括房间视图数3、任务物体数6、置信度阈值0.8及找房间权重(0.35,0.55,0.1)。迁移到新建筑需重新估计拓扑和语义类别，并适配机器人传感器。

#### 总结
核心思想：用楼层先验找对房间
1. RGB-D构建场景图与前沿。
2. VLM判断缺证据并选探索模式。
3. 语义前沿或最佳视点收集证据。
4. 拓扑距离引导未发现房间。
5. 置信度足够后回答。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.26360v1)
- [arXiv](https://arxiv.org/abs/2609.26360v1)

---

<a id='2609.25757v1'></a>
## [Minimal Recurrent Behavioral Memory for Imitation under Partial Observability](https://arxiv.org/abs/2609.25757v1)

**Authors:** Xianyao Li, Fang Xu, Rui Min, Ruitong Tian, Jing Du

**Published:** 2026-09-22

**Categories:** cs.LG, cs.IT, cs.RO

**Abstract:**

What is the least recurrent memory needed to reproduce a specified expert under partial observability? The instantaneous requirement is the conditional entropy of the expert's behavioral quotient, but recurrence must also preserve distinctions that future observations will not restore before use. We characterize this minimal recurrent behavioral memory by a compatibility relation: under transitivity its classes attain the exact minimum, while the general case is an entropy minimization over closed compatible state assignments, with exact certificates on finite instances. A sole-carrier measurement protocol separates behavioral sufficiency, excess code rate, and information carried by observations or other memory paths; experimental bit requirements refer to the induced symbolic behavioral model under the stated occupancy. Across manipulation tasks, learned code rates remain near zero- and two-bit requirements as hidden modes grow to $512$, and anticipatory memory follows a $2\to1\to0$ requirement despite zero instantaneous demand during waiting. Learning this representation remains difficult: event-agnostic future-behavior supervision yields $36/40$ sufficient seeds with one frozen configuration and improves the longest-horizon pixel setting from $0/8$ to $6/8$ sufficient held-out seeds (closed-loop success from $0.08$ to $0.57$). On unmodified community benchmarks, the protocol certifies delay-independent requirements, which sufficient codes match at mid-delay. The supervision aids commitment but can induce predictive surplus; annealing it lets imitation and rate training reduce that surplus, separating the information-theoretic target from the ability to learn it.

### 论文解读
#### 摘要翻译
论文研究部分可观测环境中的模仿学习：复现专家行为究竟需要多少递归记忆？作者提出行为记忆的理论刻画与 DIACRITIC，实现只保留对未来专家动作不可替代的信息，而非恢复完整环境状态。有限 POMDP、机器人任务和 Memory Chain 实验显示，该方法能保留必要的前瞻记忆并压低记忆速率。

#### 方法动机分析
当前观测不足以决定专家动作时，策略必须从历史保存隐藏信息。Causal state 可能保留全部未来观测分布，System ID 则恢复与动作无关的环境参数，二者都会产生冗余。论文的核心假设是：记忆只需区分那些会导致专家未来动作不同的历史；当前暂时无用、未来仍不可恢复的信息则构成前瞻记忆。

#### 方法设计详解
每一步输入离散观测、前一代码和前一动作。递归网络先提出连续残差，再以硬最近邻从代码本中选出离散代码；代码和当前观测进入动作头，输出专家动作分布。直通梯度让硬量化可训练。损失由动作交叉熵、条件记忆速率的变分惩罚和 VQ 损失组成，并加入随机未来时刻的专家动作预测监督，以克服长程依赖。理论上，行为商按当前观测及专家动作分布合并历史；传递性成立时最小速率为条件熵 H(Γ|O)，额外差值 Δ 表示前瞻记忆。实验配置包含代码容量 K=16、递归维度 32、隐藏层宽度 128。

#### 方法对比分析
与 Causal States 保留未来观测统计量不同，DIACRITIC 面向特定专家；与 Bisimulation 关注环境奖励和动力学不同，它只要求行为复现；与 System ID 恢复全部隐藏参数不同，它追求行为商的最小区分。核心创新是把前瞻记忆写成可认证的最小速率并用离散递归代码学习。因此在专家忽略世界细节的任务中更节省记忆。非传递环境不能直接使用简单等价类，需要闭合兼容状态分配。

#### 实验分析（精简版）
A' 延迟决策任务的理论需求为 2→1 bit，学习速率分别为 2.00–2.04 和 1.00 bit。Task A 中隐藏模式增至 512 时，System ID 达 8.71 bit，而 DIACRITIC 约 0.03 bit。A' 无监督成功率仅 0.14，未来行为监督后升至 0.65，任务感知监督为 0.61–0.91。优势是速率贴近行为理论下界；局限是像素闭环精度和一般非传递任务的精确最小化仍困难。

#### 实用指南
论文提供 GitHub 代码和 Python certify 工具，可认证 13 个基准。复现应先离散化观测，先检查充分性（S_Γ>0.9），再测量速率；长程任务需保留未来行为监督并在训练后退火以去除预测盈余。推理时使用当前观测与递归代码输出动作分布。迁移到连续机器人时需重新设计量化、动作输出和感知精度，不能直接套用离散等价类。

#### 总结
核心思想：按专家行为压缩记忆
1. 用观测、历史和动作递归更新离散代码。
2. 以行为兼容性定义必须保留的前瞻信息。
3. 用量化、速率惩罚和未来动作监督学习代码。
4. 先验充分性，再验证速率是否接近理论下界。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.25757v1)
- [arXiv](https://arxiv.org/abs/2609.25757v1)

---

