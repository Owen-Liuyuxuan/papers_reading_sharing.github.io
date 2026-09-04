time: 20260904

# Arxiv Computer Vision Papers - 2026-09-04

## Table of Contents

1. [Long-Horizon Consistent and Interaction-Aware World Models for Multi-Style End-to-End Driving](#2609.03225v1)
2. [RoboTok: An Internet-Scale Data Engine for Human Demonstration Retrieval and Dexterous Manipulation Learning](#2609.03199v1)
3. [BRIDGE: An Open-Source Humanoid Platform via Morphology-Control Co-Design for Physical AI](#2609.03497v1)
4. [Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language](#2609.03677v1)
5. [RoughSense: Lightweight Terrain-Induced Rover Vibration Prediction Using Point Clouds and IMU Feedback](#2609.03720v1)
6. [Rethinking 3D Noise: Learning 3D-Aware Video Priors via Optimization-Free Morphological Perturbations](#2609.03657v1)
7. [Unfold The World: Factorize 4D Properties in Reinforcing Spatial Reasoning](#2609.03729v1)

---

## Papers

<a id='2609.03225v1'></a>
## [Long-Horizon Consistent and Interaction-Aware World Models for Multi-Style End-to-End Driving](https://arxiv.org/abs/2609.03225v1)

**Authors:** Yuxuan Han, Kunyuan Wu, Liyunong Yang, Zilu Wang, Cansen Jiang, Yi Xiao, Liang Hu

**Published:** 2026-09-03

**Categories:** cs.RO

**Abstract:**

End-to-end autonomous driving has increasingly adopted world model-based reinforcement learning frameworks to improve learning efficiency through \textit{imagined rollouts}. However, existing world models suffer from three key limitations: temporal inconsistency in long-horizon imagined rollouts, inadequate modeling of ego-environment interactions, and limited adaptability to diverse driving styles. To address these challenges, we propose \textit{StyleDrive}, a world-model-based learning framework that jointly enforces long-horizon consistency, explicitly disentangles interactive traffic states, and supports multi-style policy optimization within a unified learning paradigm. First, we introduce a temporal consistency regularization that integrates historical latent states through gated cross-attention, stabilizing long-horizon imagined rollouts and mitigating error accumulation. Second, we design an explicit state disentanglement module that separates ego-relevant from ego-irrelevant interactive states, enabling more interpretable and efficient decision-making in complex traffic scenarios. Third, we enable multi-style driving behaviors through Group Relative Policy Optimization, which replaces per-step reward optimization with trajectory-wise relative advantages, reducing reward variance and supporting diverse driving styles without retraining. We evaluate StyleDrive on the Bench2Drive closed-loop driving benchmark, achieving a driving score of 88.44 (+17.08 over the previous best world model-based method) and a success rate of 66.82 (+16.58). Furthermore, we deploy StyleDrive on a real automated guided vehicle platform and demonstrate promising sim-to-real transfer capability in dynamic driving scenarios.

### 论文解读

#### 摘要翻译
论文提出 StyleDrive，一个面向多风格端到端驾驶的世界模型框架。它针对长时想象中的时间不一致、自车与环境交互建模不足、以及策略难以适配不同驾驶偏好三个问题，引入时间一致性正则化、显式状态解耦和组相对策略优化。在 Bench2Drive 闭环评测中，多模态版本取得 88.44 的驾驶分数和 66.82% 的成功率，并完成真实自动导引车部署。

#### 研究问题与动机
端到端驾驶希望直接从传感器产生控制，但模仿学习会受到分布偏移影响，直接强化学习又需要承担真实试错的安全代价。世界模型提供了折中方案：先把相机和激光观测压缩到潜空间，再在潜空间想象动作后果。然而，单步先验的自回归误差会随着想象时长累积，造成语义漂移、运动不连续和不可靠的风险判断。

现有方法还常把所有车辆、行人和道路元素作为一个环境整体处理，模型容量因而被无关对象占用，也不容易解释某个对象为何改变了自车动作。固定奖励权重则倾向于产生单一的保守或激进策略，改变驾驶风格往往需要重新训练。StyleDrive 的核心假设是，历史先验可以帮助当前预测保持一致，自车相关性可以帮助模型挑出关键交互，而对不同风格的完整轨迹做相对评价可以降低奖励尺度差异。

#### 核心方法
系统输入环视相机图像和 LiDAR 点云，输出连续的转向、油门与制动。首先，BEVFusion 把多模态信息对齐到鸟瞰视角以提取几何结构，ViT 从图像提取高层语义，随后将两路特征投影融合。融合表示进入双分支 RSSM：一个分支编码动作可控的自车状态，另一个分支编码环境和其他交通参与者。每个分支都维护确定性和随机潜变量；训练时用当前观测得到后验，推理和想象时使用先验递推，因此可以在未观测未来中滚动预测。

长时一致性正则化保存最近 5 个时间步的先验表示，并用门控交叉注意力聚合历史上下文。门控由自车状态、环境状态和动作共同决定，可以在当前先验与历史信息之间自适应权衡：当前变化明确时保留新信息，预测开始漂移时借助历史语义纠正。环境状态随后经过交叉注意力形成局部区域表示，并以自车与环境表示的余弦相似度计算语义依赖分数。依赖分数达到 0.4 的区域被视为自车相关，其余区域单独保留，策略因此能集中处理可能真正影响轨迹的车辆、行人或障碍物。

世界模型解码潜状态，预测奖励、延续概率和视觉观测。训练损失包括延续预测的负对数似然、奖励均方误差、观测重建误差，以及自车和环境分支的 KL 正则。奖励由路点进度、目的地到达、时间惩罚、驾驶平顺、碰撞和偏离组成；调整碰撞、偏离、路点和目的地权重，就可以定义保守、中庸和激进三种风格，同时保留共同的及时完成与舒适性偏好。

策略学习分为两步。第一步在冻结的世界模型中用 PPO 预训练，使策略学会利用潜状态进行闭环决策；第二步用 GRPO 微调。风格权重向量作为条件输入策略和价值网络，GRPO 则把 6 条并行轨迹的整段回报在组内标准化，以轨迹级相对优势更新，而不是过度依赖每个时间步的精确价值估计。算法使用 0.05 的裁剪系数和 0.05 的 KL 约束，想象时界为 6 步。

#### 方法对比与创新
StyleDrive 与普通世界模型的第一处本质差异是“历史先验校正”：它不只根据当前一步递推，而是用门控机制吸收近期轨迹，从而针对长时误差积累。第二处差异是“交互对象解耦”：环境并非不可解释的整体，而是按与自车状态的语义依赖拆成相关和不相关部分。第三处差异是“组内风格优化”：多种奖励配置共同形成一组轨迹，用相对回报学习偏好变化，减少为每个风格单独训练的需要。

这种组合适用于动态交通互动明显、决策需要提前想象、且用户希望调节驾驶偏好的仿真驾驶和移动机器人。它不是无条件适用于所有导航任务：若观测没有可靠的几何对齐，或任务不存在可比较的风格回报，状态分解和 GRPO 的意义会减弱。方法收益来自表示稳定、交互筛选和优化目标的配合，而不是简单增加视觉网络规模。

#### 实验结果
作者在 CARLA 的 Bench2Drive 上评估，数据约 400GB，共 1000 个片段，其中 950 个用于训练、50 个用于评估，覆盖 12 个城镇。世界模型训练 30 个 epoch，使用 8 张 NVIDIA A6000、总 batch size 64、AdamW 学习率 3×10^-4 和 weight decay 0.01；策略训练使用 6 张 A6000，学习率峰值为 1×10^-4，并退火至 1×10^-8。

多模态 StyleDrive 的驾驶分数为 88.44、成功率为 66.82%，世界模型基线 Epona 的对应结果为 71.36 和 50.24%，因此分别提高 17.08 和 16.58 个百分点。长时视觉想象在 1、3、5、10 秒的 FID/FVD 分别为 12.3/99.3、24.7/257.3、46.5/372.5、69.4/428.2；10 秒 FVD 低于 Epona 的 572.4。移除时间一致性模块后驾驶分数降到 70.11；从单风格 PPO 到加入 GRPO 的多风格训练，多样性由 0.151 增至 0.524，说明长期稳定和风格优化均对闭环表现有贡献。

#### 实用指南
复现时应保持相机与 LiDAR 的时间、坐标对齐，按 BEVFusion、ViT、双分支 RSSM、历史先验校正、语义解耦的顺序建立世界模型，然后冻结它进行 PPO 和 GRPO。论文明确给出的关键设置包括历史窗口 5、语义依赖阈值 0.4、想象时界 6、GRPO 组大小 6，以及上述训练轮数、优化器和学习率。评估应同时检查闭环驾驶分数、成功率和长时视觉质量，开放环指标不能独立代表驾驶安全。

论文展示了 Scout AGV 实车部署，使用 Jetson AGX Orin 64GB 进行实时推理，Intel NUC 11 负责数据处理，并配备 GNSS/IMU、LiDAR 和六个相机。论文没有明确给出具体推理延迟、网络层数或完整代码仓库地址，因此这些信息需要向作者或后续发布版本确认。迁移到另一种机器人时，应替换观测编码器、动作输出和奖励权重，并重新标定相关性阈值；历史校正和组相对回报机制可以作为较通用的候选组件。

#### 局限性与意义
闭环主要在 CARLA 城市场景完成，真实 AGV 的展示场景也相对有限，尚不能据此断言模型在高速公路、高密度交通、极端天气或传感器失效时同样安全。阈值 0.4、不同风格的奖励权重以及仿真与现实之间的视觉差异都可能改变结果。FID/FVD 反映想象视觉的一致性，却不等同于碰撞率或真实道路安全；论文也未充分报告部署延迟、失败类型和不同随机种子的稳定性。

尽管如此，论文的意义在于给出一条清晰的模块化路径：先用历史信息稳定世界模型，再从环境中筛选真正相关的交互对象，最后以条件化组相对优化控制驾驶偏好。真实 AGV 能针对迎面障碍、静态障碍和横穿行人执行让行、变道、停车或制动，也说明这种设计具有一定 sim-to-real 潜力，但仍需要更广泛的真实数据和安全验证。

工程上还应把潜状态预测视为风险评估工具，配合独立的碰撞监测、制动接管和动作限幅机制；风格参数不应绕过安全约束。部署前应分别检查传感器延迟、坐标标定、控制频率和异常观测下的退化行为，并用未参与训练的路线进行压力测试。

#### 总结
核心思想：历史校正关键交互，组内学习多风格。

1. 相机与 LiDAR 融合，形成自车和环境潜状态。
2. 双分支 RSSM 借助历史先验交叉注意力维持长期一致。
3. 语义依赖分离关键交互，让策略聚焦影响自车的对象。
4. 冻结世界模型先用 PPO，再用 GRPO 联合学习多种驾驶风格。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03225v1)
- [arXiv](https://arxiv.org/abs/2609.03225v1)

---

<a id='2609.03199v1'></a>
## [RoboTok: An Internet-Scale Data Engine for Human Demonstration Retrieval and Dexterous Manipulation Learning](https://arxiv.org/abs/2609.03199v1)

**Authors:** Howard Qian, Yiting Chen, Yunfei Xie, Kejia Ren, Podshara Chanrungmaneekul, Gaotian Wang, Bowen Wen, Chen Wei, Kaiyu Hang

**Published:** 2026-09-02

**Categories:** cs.CV, cs.RO

**Abstract:**

Robot learning increasingly depends on broad and diverse demonstrations, yet collecting robot data remains expensive and poorly suited to covering the long tail of real-world tasks. To address this bottleneck, we introduce RoboTok, an internet-scale data engine that, given a query human manipulation video, retrieves manipulation-relevant human demonstrations from web videos for training dexterous robot policies. Specifically, we learn a latent motion space from 3D hand trajectories expressed in estimated actor-centered reference frames. This representation enables manipulation behaviors to be compared across variations in camera viewpoint, scene appearance, and actor occlusions, while remaining compact enough for efficient search and continual indexing over internet-scale video collections. We evaluate RoboTok against existing robot-data retrieval approaches on retrieval benchmarks and downstream robot policy performance. Our results show that RoboTok retrieves more relevant manipulation demonstrations and improves downstream task success, establishing hand-pose trajectory-aware retrieval as a way to make web video a scalable and continuously growing source of supervision for robot learning.

### 论文解读

#### 摘要翻译

机器人学习高度依赖多样化的人类示教数据，但示教收集昂贵，难以覆盖长尾任务。RoboTok 提出一个互联网规模的数据引擎：根据用户输入的视频查询，从海量网络视频中检索与操作相关的人类示教视频，并用检索结果训练灵巧机器人策略。方法学习一个基于三维手部轨迹的潜在运动空间，使表示尽量不受视角、场景外观和遮挡影响，同时支持高效向量搜索。

换言之，论文并非把网络视频直接当作机器人动作序列，而是先抽取其中的手部运动，再学习一个能在大规模库中快速寻找相似动作的表示。作者希望借此把互联网视频转成可用于机器人探索的结构化运动监督。

#### 研究问题与动机

机器人示教通常依赖人工遥操作、动作捕捉或专门数据采集。这些方法质量较高，但成本和覆盖范围有限，即使 Open X-Embodiment 等大型机器人数据集也只包含现实操作行为的一部分。互联网视频数量巨大，蕴含拧、推、拉、抓取和多指协调等行为，却没有统一的机器人动作标签，而且拍摄角度、背景、人物、遮挡和执行速度各不相同。

真正困难之处是区分“视觉上相似”和“运动上可借鉴”。两个视频可能都显示一个瓶子，却有不同的手指路径；反过来，同一种操作可以由不同的人在不同房间、视角和速度下完成。RoboTok 的假设是，经过坐标规范化的三维手部轨迹更接近灵巧操作的几何结构，而动态时间规整（DTW）可以消除动作快慢不同带来的时间错位。问题边界是：该假设主要适用于手部可见、动作较短且手部运动足以描述任务的场景，不能自动替代物体、力和语言语义建模。

#### 核心方法

RoboTok 的输入是一段人类操作视频，输出是大规模索引库中按动作相似度排序的若干视频片段。第一步是数据筛选：从 Action100M 中提取 4–8 秒片段，用 Lucas–Kanade 光流保留近静态相机，并限制一段视频中最多一只左手和一只右手，从而降低相机运动和多人交互造成的混淆。

第二步把像素变成三维轨迹。WiLoR 以 5 fps 检测每只手的 21 个三维关节；MoGe-2 估计度量深度，把关节放到度量相机坐标系；遮挡或检测失败的帧由 HaWoR 补全。得到的不是单帧姿态，而是随时间变化的手部序列，因此可以表达手指和腕部的协调运动。

第三步是以人为中心的规范化。相机坐标会随第三人称拍摄位置改变，作者训练一个轻量 torso-frame estimator，仅从手腕帧预测示教者的静态躯干坐标系，再把轨迹转换到 actor-centered 的 egocentric 坐标。直觉是，同一个拧盖动作在人体躯干坐标下具有较稳定的相对方向和幅度，即使拍摄者换了位置，运动描述仍可比较。

第四步学习检索嵌入。轻量 cross-attention 编码器接收规范化的时空手部姿态，加入位置编码并聚合序列，输出单位超球面上的 d 维向量 Γ(x)。监督来自 DTW，而不是手工动作类别。论文用

DTW(x_i,x_j)=min_π Σ_(t,u)∈π ||x_i^t−x_j^u||₂

衡量两段轨迹在时间拉伸或压缩后的最小累积距离，并以长度归一化负 DTW 成本定义相似度 s(i,j)。如果 j 比 k 更接近 i，就要求嵌入内积 Γ(x_i)·Γ(x_j) 大于 Γ(x_i)·Γ(x_k)。每个锚点组包含 1 个锚点、2 个正样本和 1 个边界负样本；49 组构成 batch size 196，DTW 排名前 20 的邻居用于正样本。联合损失 L_set+λL_rank 同时学习相似集合边界和集合内部排序。

论文给出的训练规模是 100,000 段 Action100M 片段，验证集 10,000 段；跨域评估使用 831 段带传感器级三维手部标注的 AssemblyHands。推理时全库轨迹先离线编码并放入向量索引，查询视频只需要一次前向编码和余弦近邻搜索，无须对查询与每个候选重新计算 DTW。这一设计把 DTW 保留为训练 oracle，同时将互联网规模检索的在线成本降为向量搜索。

#### 方法对比与创新

FlowRetrieval 主要依靠光流轨迹，容易受到相机和图像运动影响；HAND 使用二维手路径并结合视觉相似性过滤，仍可能把外观相似但动作不同的片段排在前面；STRAP 使用视觉基础模型特征并做 DTW 对齐，但不是专门为三维手部几何和机器人灵巧操作构造。RoboTok 的关键差异是把检索目标从“场景或语义相似”转为“规范化三维手运动相似”。Random 则提供无动作先验的参考。

其创新可以理解为三个相互衔接的环节。首先，用 WiLoR、MoGe-2 和 HaWoR 从无标签视频构造三维手部运动数据；其次，从腕部轨迹估计躯干参考系，削弱第三人称视角差异；最后，以 DTW 产生的连续排序监督训练可索引嵌入，而非把动作粗略分成有限类别。适用场景是手部运动、速度变化和视角变化决定成功与否的任务；若关键差异来自物体几何、接触力或任务语言，必须加入额外表示。

#### 实验结果

在 Action100M 验证集的 top-20 检索评测中，RoboTok 的 mAP@20 为 0.3531，而 STRAP 为 0.0071；RoboTok 的 Recall@20 达到 0.996。top-20 平均 DTW cost 为 1.333，接近直接 DTW oracle 的 1.145，并优于随机检索的 4.776。这些结果说明学习到的向量邻居确实更接近三维运动相似度，而不是只在外观上相像。

跨到 AssemblyHands 时，RoboTok 的 mAP@5 为 0.2614，STRAP 为 0.1330，显示一定跨域能力。下游验证在 VTDexManip 灵巧操作仿真中进行：作者把检索示范状态的负 k-NN 距离用作 PPO 奖励。原始任务的 6 项任务中，RoboTok 有 5 项超过 VT-JointPretrain，seen 和 unseen 任务平均成功率提升分别为 7.45% 和 5.83%。在去除手工稠密奖励、允许不受限三维运动的 HARDER 设定下，BottleCap Turning 为 77.3%（HAND 59.5%），Faucet Screwing 为 44.8%（HAND 6.8%），Lever Sliding 为 79.3%（HAND 19.5%）。因此检索质量不仅体现在离线指标，也能为稀疏反馈下的策略探索提供方向，但真实机器人效果尚未由这些实验直接证明。

#### 实用指南

复现的基础是先保持论文的数据处理约束：4–8 秒片段、近静态相机、最多一只左手和一只右手、5 fps 采样及每只手 21 个关节。需要运行 WiLoR 进行手部三维重建，用 MoGe-2 获得度量深度，并用 HaWoR 处理缺失姿态；随后训练或使用 torso-frame estimator，把轨迹放入 actor-centered 坐标。训练规模可参考 100,000 段训练片段、10,000 段验证片段和 batch size 196；论文未完整给出学习率、训练轮数或硬件配置，复现时不应自行补写这些超参数。

论文提供项目主页 robotok-engine.github.io/，并列出所使用的 WiLoR、MoGe-2、HaWoR 等组件；完整代码、权重和 Action100M 索引是否公开，应以主页实际提供内容核对。工程部署应将轨迹提取、规范化、编码和索引建立放到离线阶段，在线只编码查询并做余弦搜索；评估时同时报告 top-k 检索指标和 DTW 行为质量，避免只用图像相似性判断。

迁移到新数据集或新机械手时，需要重新处理坐标系、重建索引，并确认人类 21 关节运动如何映射到机器人可执行的手指和腕部自由度。移动相机需要额外的自运动补偿；手部经常被遮挡时需要更鲁棒的补全或不确定性建模；若物体尺寸、接触和力是任务关键，则应在手轨迹之外加入物体和接触特征。下游 PPO 奖励还必须适配目标仿真器或真实机器人状态接口。

#### 局限性与意义

系统依赖近静态相机筛选，因而会漏掉许多手持相机和第一人称视频。WiLoR、深度估计和姿态补全的误差可能累积并改变轨迹排序，腕部推断躯干参考系也可能在极端人体姿态下失效。当前索引的主体是手部运动，没有完整整合物体三维几何、接触状态、力反馈和语言任务条件；检索到的人类行为也不保证直接适合某一种机械手。

此外，跨 AssemblyHands 的检索结果不能等同于真实部署泛化，下游实验主要在 VTDexManip 仿真中完成，第三人称视频到机器人观测之间仍有 embodiment 和视觉域差异。论文的意义在于提供一种可扩展的数据引擎：把不断增长的网络视频转成动作候选和探索先验，降低寻找示教的成本；它不是端到端通用策略，仍需要任务筛选、动作映射、真实闭环验证以及对数据偏差的审查。

#### 总结

核心思想：用规范化三维手运动检索示教。

全文 pipeline 可速记为：

1. 从网络操作视频筛选短片，抽取并补全 21 关节三维轨迹。
2. 用腕部预测躯干参考系，把不同相机视角统一到人体中心。
3. 用归一化 DTW 监督 cross-attention 编码器学习运动排序。
4. 离线编码全库并建立余弦索引，在线一次编码取得相似示教。
5. 用检索示范距离构造 PPO 奖励，帮助灵巧策略探索困难任务。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03199v1)
- [arXiv](https://arxiv.org/abs/2609.03199v1)

---

<a id='2609.03497v1'></a>
## [BRIDGE: An Open-Source Humanoid Platform via Morphology-Control Co-Design for Physical AI](https://arxiv.org/abs/2609.03497v1)

**Authors:** Jianren Wang, Letian Qian, Zikai Wang, Weiwei Wu, Junjie Zong, Abhinav Gupta, Deepak Pathak

**Published:** 2026-09-03

**Categories:** cs.RO, cs.AI

**Abstract:**

Developing humanoid robots capable of leveraging human behavioral data is essential for general-purpose embodiment, yet conventional development remains bottlenecked by a decoupled paradigm that isolates hardware design from whole-body control. This approach leads to suboptimal systems that compromise human-like fluidity and agility. To bridge this gap, we introduce a data-driven morphology-control co-design framework that optimizes humanoid morphology for human-like movement. To quantify morphological fidelity, we also introduce a novel metric that jointly considers kinematic retargeting fidelity to human motion and dynamic tracking performance. Our framework achieves state-of-the-art (SOTA) performance across all metrics compared to baseline humanoids (Bumi, K1, and Toddlerbot). Finally, we realize this design in Bridge, an open-source, 88cm-tall humanoid platform released alongside its control policy. We demonstrate that Bridge captures human motion data with superior fidelity, exhibiting exceptional performance across foundational locomotion, robust balance, and highly dynamic maneuvers. Videos and open-source materials: https://sites.google.com/view/bridgerobot.

### 论文解读

#### 摘要翻译
论文提出 BRIDGE，一个通过形态—控制协同设计支持 Physical AI 的开源人形机器人平台。作者指出，传统开发把硬件形态和全身控制解耦，形态由启发式规则确定，控制器随后补偿关节拓扑、比例、活动范围和执行器能力的缺陷，因而难以稳定复现人类运动。本文提出数据驱动的协同设计框架，以运动学重定向保真度和动态跟踪性能共同定义人形相似性。最终 BRIDGE 高 88 cm、重 13 kg，具有 21 个主动自由度；与 Bumi、Booster K1 和 Stanford ToddlerBot 比较时，在所报告的人形相似性指标上取得最佳结果。

#### 研究问题与动机
研究问题是：在高度小于 90 cm、电池和执行器封装受限的前提下，怎样选择关节拓扑、轴位置和执行器，使小型人形机器人不仅几何上像人，也能在闭环控制中完成来自人类数据的平衡、日常和高动态运动。人类行为数据规模大、动作丰富，但机器人若缺乏相容的物理形态，就无法直接利用这些数据。

传统流程的痛点包括机械设计依赖手工经验、执行器常在后期才被考虑，以及固定硬件迫使控制算法补偿不可改变的结构缺陷。论文的核心假设是，运动学误差低并不等于动态可行；应把运动学重定向误差、策略执行后的动态跟踪误差、运动覆盖率和执行器峰值利用率放入同一迭代闭环。

#### 核心方法
框架从保留人类主要关节的 23-DoF 模型开始。输入是 SMPL 派生的人体运动和候选机器人形态；先做自由度压缩，在腰部三种双自由度组合中比较运动学重定向误差。Roll+Yaw 的误差为 0.02570，优于包含 Pitch 的候选，因此移除腰部 Pitch，以满足小于 90 cm 的高度目标并为电池留出空间；在此基础上再比较只保留腰部 Roll 或只保留 Yaw 的 21-DoF 候选，最终因动态跟踪误差 E_dyn=0.02115 选择腰部 Yaw。

然后进行执行器感知的实例化。每个关节从候选集合中先分配体积最小的执行器，并纳入包络、安装间隙、质量、惯性、减速比和扭矩—速度曲线。设计使用实验校准的最大扭矩，而非标称峰值。复合关节把 Pitch 轴锚定在解剖学关节中心，并沿肢体方向平移 Roll/Yaw 轴以容纳电机，同时约束相邻执行器间距和碰撞。上肢使用约 10 Nm，髋 Pitch 与膝 Pitch 约 55 Nm，其余关节约 25 Nm；膝 Pitch 示例执行器质量 0.450 kg、厚度 58 mm、减速比 36、峰值扭矩 55 Nm。

运动评估阶段先使用共享的 21-DoF 基础策略，再针对形态和参考运动微调策略。论文说明控制策略训练采用 SONIC，并以 BeyondMimic 等工作作为相关运动跟踪基础；在 MuJoCo 中执行 LaFAN1 与 bones_seed 组成的参考动作集。动作只有在稳定完成、完成度达标且动态误差不超过阈值时才计入运动覆盖率。失败后统计每个关节的峰值利用率（峰值扭矩除以校准上限），若某关节反复饱和且解除该瓶颈后动作可成功，就局部升级执行器，重新计算布局和惯性、重建 URDF 并重训。论文未报告 batch size、学习率、训练轮数、GPU 型号或固定推理延迟，复现时不能擅自补写。

这个闭环的关键不是把所有电机都换成更大规格，而是把失败转化为结构修改的证据：运动学阶段回答“姿态能否映射”，动力学阶段回答“实体能否跟上”，利用率分析则回答“具体哪一个关节限制了成功”。因此每次迭代都同时更新形态参数、惯性模型和策略，而不是在控制器中隐藏硬件不足。

最终人形相似性为 S_HL=exp[-(0.5E_kin+0.5E_dyn)/0.05]，将几何保真与动态可行性等权合并。评估还使用 MPJPE、MPJVE、RootVelErr、MPKPE 和成功率；推理时因此不能只检查关节角是否接近参考，还要检查策略执行后的状态轨迹是否稳定。

#### 方法对比与创新
BRIDGE 与“先定机械结构、再训练控制器”的解耦方法不同，把形态候选筛选、真实执行器封装、策略微调和失败分析串成闭环。第一项创新是用运动学重定向和动态跟踪共同构成 S_HL，而不是只以人体比例或静态姿态误差选型；第二项创新是用运动覆盖率判断结构是否能执行动作；第三项创新是依据执行器饱和定位局部瓶颈，只升级必要关节并回写机械与控制模型。

该方法适合受尺寸、重量和成本约束、又要从人类运动数据学习的小型人形平台。它仍依赖工程师定义候选拓扑、执行器库、动作数据和成功判据，并非任意任务的自动结构生成器。相对 Bumi、K1 和 ToddlerBot，BRIDGE 还以开源硬件、设计和全身控制策略降低了 Physical AI 实验的进入门槛。

#### 实验结果
论文在 MuJoCo 中将动作分为 Balance、Highly Dynamic 和 Daily Motion，并与 Bumi、K1、ToddlerBot 比较。BRIDGE 的 E_kin=0.0260、E_dyn=0.0384、S_HL=0.5252；基线 S_HL 为 Bumi 0.4321、K1 0.4198、ToddlerBot 0.3883。整体运动跟踪成功率为 94.83%，MPJPE=0.0711、MPJVE=0.5167、RootVelErr=0.1671、MPKPE=38.43；三类动作成功率分别为 95.00%、94.50% 和 94.99%，高动态类别较最强基线高 4.70 个百分点。

形态迭代揭示了物理约束的重要性：初始方案 S_HL=0.7558、成功率 98.27，但加入执行器限制后成功率降至 44.63%；经过失败定位和执行器升级，后续方案恢复到 94.83%。真实 BRIDGE 完成单腿平衡、Charleston 舞、后空翻和遥操作序列；收敛比较中，其 value-function loss 下降最快且最终值最低。上述证据支持小型平台上的运动跟踪和物理可行性，不等同于重载操作验证。

#### 实用指南
论文明确将 BRIDGE 硬件、设计和全身控制策略作为开源平台发布，但正文没有给出可核验的代码仓库地址、依赖版本、完整训练超参数或计算设备。因此复现者应重点保留 SMPL 到机器人关节的重定向流程，使用 LaFAN1 与 bones_seed 动作，按实测曲线校准扭矩上限，建立含质量和惯性的 URDF，并在 MuJoCo 中同时报告 E_kin、E_dyn、运动覆盖率和执行器利用率。摘要给出平台约 88 cm、13 kg；对比表写作约 0.8 m、12.5 kg，复现报告应说明采用的口径。

迁移到另一台机器人时，需要替换关节拓扑、连杆尺寸、执行器包络与扭矩—速度曲线，并重新微调控制策略和成功阈值，不能直接复制 21-DoF 轴布局。更大平台可能扩大工作空间和负载，但要重新处理质量、惯性、功率和控制频率。只优化 MPJPE 可能得到姿态相似却无法稳定执行的结构，工程评估应保留动态误差与动作成功率。

实际复现还应区分“候选形态筛选”和“实体验收”两个层次：先在相同动作集上对候选 URDF 做运动学重定向，随后以真实扭矩上限和碰撞约束运行控制评估，最后才进行实体动作测试。论文未说明完整的软件依赖、训练轮数和随机种子，故不同实现的绝对数值可能存在差异；更稳妥的比较方式是沿用同一动作划分、同一成功判据和同一指标定义。

#### 局限性与意义
作者承认平台目前完全依赖旋转电机；腱驱动可能缩小关节体积并把电机质量移出肢体。88 cm 的小型化限制工作空间和负载能力，复杂操纵能力不能由平衡、舞蹈或后空翻演示外推。论文主要验证运动跟踪和动态动作，没有建立重载操作、长期可靠性、能耗或复杂现实环境泛化的证据。

此外，仿真策略、实测执行器曲线与真实装配误差之间仍有 sim-to-real 风险，失败判定和动作集合也会影响运动覆盖率与 S_HL。BRIDGE 的意义在于把“像人”从外观或静态比例判断改成可测的几何—动力学联合问题，并提供成本约 1.5K 美元、可公开复用的实体平台，推动硬件设计、全身控制和人类运动数据学习共同迭代。

#### 总结
核心思想：形态与控制共同塑造类人运动。

1. 从 SMPL 运动和 23-DoF 人体拓扑压缩出满足高度约束的候选形态。
2. 把真实执行器包络、校准扭矩和关节轴偏移写入 URDF，形成可制造实例。
3. 用基础策略和按动作微调在 MuJoCo 中检验动态跟踪，而非只看静态重定向。
4. 从执行器饱和失败定位瓶颈，局部升级后重建模型并重训，最后用 S_HL 和运动覆盖率筛选。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03497v1)
- [arXiv](https://arxiv.org/abs/2609.03497v1)

---

<a id='2609.03677v1'></a>
## [Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language](https://arxiv.org/abs/2609.03677v1)

**Authors:** Julian Truetsch, Felix Hauser, Christoph Stiller, Frank Bieder

**Published:** 2026-09-03

**Categories:** cs.CV, cs.CL, cs.LG, cs.NE, cs.RO

**Abstract:**

Understanding the composition of large-scale autonomous driving datasets is essential for safety, robustness, and reliable operation across domains. For example, domain shift between locations could lead to the operating environment being misaligned with the training data, resulting in potentially dangerous performance degradation. Yet, existing data analysis pipelines largely rely on metadata, predefined labels, or manual inspection, which provide limited semantic insight or do not scale. This paper studies set difference captioning: given two subsets of images, the goal is to produce a natural-language hypothesis describing differences between the target and reference set. Building on a two-stage formulation, we adapt the method to autonomous driving by focusing on object-centric patches derived from object detection, which simplifies aggregation and enables attribution of differences to specific object instances or categories. To evaluate this setting in-domain, we introduce a new benchmark, AD-Diff Bench. Low-concentration experiments assess the suitability of set-difference-captioning approaches to sparse, real-world differences. We restrict our experiments to open-weight models to support reproducibility and ease of deployment. The proposed benchmark and analysis provide a step towards practical, human-interpretable dataset introspection for autonomous driving datasets. Our implementation and benchmark dataset are available at https://github.com/KIT-MRT/AD-Diff

### 论文解读

#### 摘要翻译

论文研究如何让研究者理解大规模自动驾驶数据集中的分布差异。给定目标图像子集和参考子集，方法不是只返回一个数值距离，而是生成“目标集相对于参考集更常见什么”的自然语言描述。作者提出对象中心的图像补丁处理和 proposer-ranker 两阶段架构，并构建 AD-Diff Bench，用于评估系统能否发现数据偏置、域偏移以及现实中稀疏但重要的差异，例如救护车、施工人员或车辆状态。

#### 研究问题与动机

自动驾驶数据的地点、天气、道路设施、传感器和交通参与者外观不断变化。模型若在一个分布上训练、在另一个分布上部署，可能出现泛化下降甚至安全风险。元数据和预定义标签能回答数量、类别等结构化问题，却难以表达“某个子集里的车辆更常出现在某种状态”或服饰、上下文等长尾语义；人工逐张检查又无法随数据规模增长。

既有差异描述工作多比较两张图像，把它直接扩展到两个集合会产生大量成对比较，而且通用集合差异基准对道路场景覆盖不足。论文的核心假设是：少量样本适合让生成模型提出可读的候选解释，而让候选在完整集合上重新验证，能够聚合分散的弱证据，减少一次抽样偶然漏掉差异的风险。

#### 核心方法

完整输入输出流程是：目标集 A 与参考集 B → 检测/标注对象 → 提取对象中心 patch → 少量样本提出假设 → 全量集合排序 → 输出最能区分 A、B 的自然语言差异。对象框可来自预训练 2D 检测器或真值标注；裁剪框相对原框扩大 50%，保留对象周围道路环境，并用红色矩形标记目标位置，使 VLM 在拥挤交通画面中不被其他参与者带偏。

提议阶段每轮从 A 和 B 各采样 20 张图像并生成 10 条候选，独立重复 3 轮。image-based proposer 直接观察多个 patch；caption-based 先得到单图描述，再让语言模型汇总集合差异；feature-based 将图像缩放到 224×224 后提取嵌入，利用两集合特征均值之差的方向帮助解码器生成描述。这样生成模型只处理小批候选，避免把全量图片同时放进上下文。

排序阶段把每条候选假设 h 放回完整 A、B。SigLIP 2 Giant 计算图像 x 与文本 h 的嵌入余弦相似度 R(x,h)，把该分数当作区分两个集合的二分类证据，并以 AUROC 衡量候选的集合区分能力；最终优先保留最准确的假设。生成式组件采用 Qwen3-VL-30B-A3B-Instruct。论文给出了上述采样、模型和 224×224 输入设定，但没有给出训练 batch size、学习率或 epoch，复现时应明确区分已报告的推理配置与未报告的训练超参数。

这种组合同时施加了可读性和可检验性约束：候选必须是人能够理解的句子，又必须在 A 与 B 的图文相似度分布中体现稳定方向。因此，ranker 并非寻找最华丽或最像某一张图的描述，而是在验证一句话能否真正区分两个集合。这一点对数据审计尤其重要，因为输出可以进一步追溯到集合构造、采样策略和对象属性，而不只是一个不可解释的嵌入距离。

#### 方法对比与创新

单阶段基线让 VLM 一次处理每组 100 张图像，并同时生成和排序描述。它流程短，却容易受到上下文长度、注意力分散和稀疏事件的影响。两阶段方法把“开放式提出语言”与“对全量样本作可计算验证”分开：proposer 负责覆盖可能的语义，ranker 负责检查这些语义是否真的在两组数据中形成稳定区分。对象中心化则将复杂道路场景拆成同类目标单元，降低无关目标带来的噪声。

AD-Diff Bench 也把道路数据诊断纳入集合差异描述评测。它包含 180 对 Bing 网页检索集合、60 对从 KITTI、nuImages 和 Waymo Open Dataset 细粒度标注筛出的集合，以及 80 对用 CLIP 相似度筛选的对象 patch 集合。网页集合适合测试较明显的语义差异，标注筛选集合更接近车载低分辨率和细微状态差异，CLIP 集合则考察视觉外观层面的难例。方法适合数据筛选、偏置审计和部署前域差异诊断，但不替代安全认证、因果分析或精确统计检验。

三个 split 的难度设计也说明了方法要面对的证据层次：网页检索图像通常具有较强的可见类别线索，标注筛选可以把差异压缩到对象状态，CLIP 筛选则可能只保留外观相近而语义不明显的 patch。因而不能只用网页结果推断真实车载场景的可靠性；更合理的使用方式是把不同 split 看成从明显差异到细粒度差异的分层压力测试。

#### 实验结果

论文以 Acc@1 和 Acc@5 评价候选排名，Acc@1 表示第一名假设正确的比例。两阶段 image-based 方法在 web-scraped、annotation-filtered、CLIP-filtered 三个 split 的 Acc@1 分别为 0.73、0.56、0.64；caption-based 方法分别为 0.70、0.60、0.63。feature-based 只有 0.33、0.20、0.41，说明仅依靠视觉特征提出自然语言解释，较难覆盖细粒度集合语义。image-based 在网页 split 的 Acc@5 还达到 0.88。

对照单阶段 image-based，三个 split 的 Acc@1 为 0.64、0.53、0.49；两阶段方法因此在网页和 CLIP 筛选集合上提升明显，在最难的标注筛选集合上也略有优势。标注筛选难，部分因为车载图像分辨率更低，且“停靠”与“停车且有驾驶员”这类状态差异很细。稀释实验用 c=n_S/(n_S+n_D) 表示含差异样本 n_S 在差异样本与干扰样本总数中的浓度；c 约在 0.5 之前性能相对稳定，进一步下降后准确率快速降低，支持全量 ranker 聚合弱证据的设计。gpt-oss-120b 的 0/0.5/1 语义等价评分与人工判断一致率为 80.3%，平均绝对误差为 0.104，但自动评审仍应作为辅助证据。

#### 实用指南

复现的最小配置是准备目标集、参考集和对象框，统一进行 50% 框扩展及红框标记；每轮从两集各取 20 个 patch，运行 3 轮、每轮产生 10 条候选，再在全量集合上用 SigLIP 2 Giant 排序。若使用生成式 proposer，采用 Qwen3-VL-30B-A3B-Instruct；feature-based 分支需要将图像缩放到 224×224。应记录集合规模、对象来源、采样随机种子与检测质量，因为它们会影响稀疏差异的可重复性。

代码和数据已由 KIT-MRT/AD-Diff 开源，开放权重模型使隐私敏感的车载数据可以考虑离线部署。论文没有报告训练 batch size、学习率、epoch、GPU 显存或端到端延迟，因此这些项目不能凭空设定为论文结果。迁移到其他领域时，可沿用“对象裁剪—少样本假设生成—全量验证”骨架，替换对象定义、检测器、领域提示词和集合构造。数据极大时可采用分层采样或近似排序；安全应用仍应由人工和统计检验复核。

#### 局限性与意义

方法首先受对象检测或标注质量限制：错误的框会把背景或错误目标送入 VLM。网页预训练模型与低清晰度车载传感器图像之间存在域差距，模型对极细语义仍不稳定。完整集合排序能整合弱信号，却需要更多推理计算；当差异极端稀疏时，例如海量图像中仅有极少数救护车，浓度低于约 0.01 后，候选生成和相似度统计都可能错过它。自然语言描述应理解为数据审计线索，而不是因果解释、检测结果或安全保证。

论文的重要意义在于建立了一个面向自动驾驶的“可解释集合差异”任务和基准，连接对象级视觉理解、开放权重 VLM 与集合级评估。它让数据工程师能更快定位训练/部署偏置，也为跨地点、天气和传感器迁移提供可读诊断。未来若加入主动采样、专门的长尾事件检索、车载域适配 VLM，并把语言描述与统计置信度结合，低浓度差异的召回和可信度仍有提升空间。

#### 总结

核心思想：少量提议、全量验证集合差异。

通俗 pipeline 速记：

1. 从目标集和参考集检测同类对象，扩大边界框并保留环境上下文。
2. 从两组各抽少量 patch，让 VLM 产生多条自然语言差异假设。
3. 用 SigLIP 2 Giant 在完整两组样本上计算图文相似度与区分能力。
4. 排序并输出最可信的偏置、域漂移或长尾差异解释，再由人和统计检查复核。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03677v1)
- [arXiv](https://arxiv.org/abs/2609.03677v1)

---

<a id='2609.03720v1'></a>
## [RoughSense: Lightweight Terrain-Induced Rover Vibration Prediction Using Point Clouds and IMU Feedback](https://arxiv.org/abs/2609.03720v1)

**Authors:** Gabriel Manuel Garcia, Stephanie Aravecchia, Miguel Angel Olivares-Mendez

**Published:** 2026-09-03

**Categories:** cs.RO

**Abstract:**

Autonomous navigation in space requires reliable terrain assessment for safe operations, especially in underground environments with limited communication, computing resources, and power budget. This paper presents a lightweight method for real-time vibration-aware traversability mapping using a Light Detecting And Ranging (LiDAR) point cloud and Inertial Measurement Unit (IMU) measurements. An initial vibration proxy is estimated from terrain geometry by applying Random sample consensus (RANSAC) to local point-cloud patches produced by a Simultaneous Localisation And Mapping (SLAM) algorithm. In parallel, the IMU provides local observations of the vibration experienced by the rover during traversal. The point-cloud-based prediction is then corrected online using Recursive Least Squares, allowing the system to adapt the geometric estimate to the measured rover response. The approach is evaluated in a lunar analogue environment, an outdoor field, and an underground mine.

### 论文解读

#### 摘要翻译

RoughSense 面向月面、火星、矿井等光照不足、通信受限且计算资源紧张的环境，提出一种轻量级的地形诱导振动预测方法。它把 LiDAR 点云提供的几何信息与 IMU 测得的车体实际振动结合起来，并通过递归最小二乘法（RLS）在线修正预测模型，实时生成带有振动感知的可通行性地图。核心目标不是简单判断“能不能走”，而是提前估计一块地面会给车体、传感器和导航系统带来多大颠簸。

#### 研究问题与动机

在极端环境中，持续冲击可能造成机械疲劳、传感器失稳或任务失败，因此路径规划需要把平稳性和振动风险纳入代价。传统 LiDAR 几何方法具有前视能力，可以在车辆尚未驶入之前标记粗糙地形，却无法直接表达悬挂、轮地接触、速度和载荷等动力学因素；反过来，IMU 能够测量真实的车体响应，却只能告诉系统已经驶过的区域，存在时间滞后。两者各自可靠但信息不完整。

论文的核心假设是：局部地面相对于拟合平面的几何残差可以作为振动先验，但几何粗糙度到实际振动的映射会随机器人平台、行驶速度和地表材料变化。因此，系统应让点云负责“提前预测空间风险”，再用 IMU 负责“校准车辆自身对该风险的响应”。这种假设既避免把 IMU 降级为事后评估，也避免依赖规模很大的端到端学习模型。

#### 核心方法

系统的输入是 Livox MID-360 LiDAR 点云、由 RTAB-Map SLAM 提供的位姿、IMU 加速度以及机器人线速度，输出是经校正的全局或局部振动成本图。完整 pipeline 可概括为：点云与位姿聚合 → 局部栅格化 → 每格 RANSAC 平面拟合 → 计算几何振动分数 → IMU 去机械噪声并速度归一化 → 滑窗计算实测振动 → RLS 学习映射 → 更新前方成本图。

第一条是外感受预测支路。点云被投影并聚合到机器人周围的局部栅格，每个栅格内用 RANSAC 拟合地面平面。设点到平面的残差为 (r_i)，有效点数为 (N)，几何代理值为 
\(\hat V_{pc}=\sqrt{\sum_{i=0}^{N}r_i^2/N}\)。残差越大，表示该单元相对局部平面越不规则；经过预设 min-max 标定后转为 [0,1] 的分数，并写入前方振动代价。点云支路的意义在于把风险投射到尚未驶入的空间。

第二条是本体感受观测支路。作者先通过频谱实验识别 Leo Rover 约 27.5 Hz 的电机/机械固定振动峰值，再用陷波滤波器抑制它，避免把平台自身噪声误当作地形冲击。随后取加速度向量的 L2 范数，并按 (N_{imu}=\hat N_{imu}/max(v,\epsilon)) 做速度归一化；其中 (epsilon) 用来避免低速或停车时除零。系统在 200 Hz IMU 信号上使用 (k=100) 的滑动窗口，约对应 0.5 秒，计算归一化信号相对窗口均值的 RMSE，形成观测分数 (\hat V_{imu})。

第三步是在线校正。RLS 用线性模型 (V_{corrected}=\alpha V_{pc}+\beta)，把特征向量写成 ([V_{pc},1]^T)，参数为 ([\alpha,\beta]^T)。每获得新的 IMU 观测，算法根据预测与观测的残差计算增益并递归更新参数和协方差，再把新映射应用于成本图。遗忘因子设为 (lambda=0.995)，既保留历史稳定性，又允许车辆在新地形中适应。实现还包含残差门控、参数裁剪和协方差膨胀，以降低异常观测对在线模型的破坏。论文没有报告离线训练 epoch、batch size 或神经网络训练过程；这是一个在线参数估计流程。

#### 方法对比与创新

与纯点云粗糙度方法相比，RoughSense 不把几何残差直接等同于车体振动，而是承认“同样的地面形状”在不同车辆和速度下可能导致不同响应。与纯 IMU 方法相比，它保留了前视预测能力，使路径规划器可以在进入风险区域之前调整路线或速度。与端到端神经网络相比，RLS 只有比例、偏置及协方差等少量状态，更新可解释、计算开销小，也不要求大型带标签数据集和反向传播。

创新点在于形成“几何先验 + 物理反馈”的闭环：点云决定风险在哪里，IMU 学习风险对当前平台意味着什么，RLS 把这种关系快速反馈到地图。它特别适合资源受限的轮式探测车和特种作业机器人；若平台动力学差异较大，也可以保留总体框架而重新辨识固有频率、标定范围和映射参数。

#### 实验结果

作者在 Leo Rover 上搭载 Livox MID-360 LiDAR 和 IMU，并在 NVIDIA Jetson Orin 上仅使用单核 CPU 运行。测试覆盖 Lunalab 月壤模拟环境、Symphony Lake 户外碎石岸和 Walferdange 地下矿井。验证时把预测到某栅格的结果与机器人未来实际驶入该栅格时的 IMU 振动观测对齐，并比较未校正的点云预测与 RLS 校正结果。

相对纯点云预测，RLS 使三个场景的平均预测误差分别降低 46%、73% 和 25%；其中户外碎石地的改善最大，说明不均匀轮地接触造成几何残差与实际冲击偏差较大时，反馈尤其有价值。实时性方面，点云振动预测约耗时 42.39 ms，RLS 修正约 234 μs，总周期约 44.43 ms，RLS 只占约 0.53%，表明在线适应几乎不是主要瓶颈，主要成本来自点云处理。

#### 实用指南

论文给出了 GitHub 实现（RoughSense 的 ISPARO2026 分支）。复现时首先要准确标定 LiDAR、IMU 与 SLAM 坐标，确认位姿不会把点云错误投影到成本图；其次应按目标平台测量机械振动频谱，不能未经验证地照搬 27.5 Hz 陷波频率。点云栅格要在空间细节和每格有效点数之间折中，并记录用于 [0,1] 归一化的 min/max；论文只说明这些边界依据“勉强可通行区域”手工设定，没有公开每个环境的具体数值。

实现参数可从论文报告的设置开始：IMU 频率 200 Hz、窗口长度 (k=100)、RLS 遗忘因子 0.995，并在速度归一化时设置非零 (epsilon)。部署前应在已知路面上检查 RMSE 分数和真实冲击的对应关系，尤其注意低速时除法放大噪声。迁移到其他轮式平台通常需要重新测固有频率、速度范围、栅格分辨率和标定边界；迁移到腿式机器人则需把轮地振动特征替换为适合足端/机身动力学的先验。论文未提供完整数据划分、训练配置或跨平台零样本保证，因此复现重点是传感器同步、时空对齐和在线标定，而非寻找未报告的离线训练超参数。

工程上还应把预测时刻与实际驶入时刻严格对齐，并在地图更新周期内检查时间戳和速度单位；否则误差可能来自同步偏差而非模型本身。建议先用短距离、速度稳定的路线验证滤波和分数单调性，再逐步放开速度与地形范围。

#### 局限性与意义

前视质量受点云密度和定位质量限制：远处点云稀疏时，几何平面和残差都不稳定；矿井长距离预测中，论文观察到校正收益几乎消失。栅格过大可能抹平小尺度障碍，过小则点数不足、RANSAC 拟合不可靠。SLAM 漂移还会使历史点云与当前地图错位。系统的线性映射 (\alpha V_{pc}+\beta) 对复杂悬挂、速度、载荷和材料变化的表达能力有限，异常 IMU 观测也可能影响在线更新；因此实现中的门控和裁剪并不能替代更丰富的动力学模型。

尽管存在这些边界，论文展示了一个很有工程价值的范式：用可解释的几何量提前规划，用廉价的物理反馈适应具体车辆。三个不同环境中误差均下降，且 RLS 仅需 234 μs，说明方法在计算受限平台上具有实际部署潜力。未来可按速度或地形类别维护多组映射，或采用能表达非线性的校正器，同时结合不确定性估计处理稀疏点云。

#### 总结

核心思想：IMU 在线校准点云振动预测

通俗 pipeline 速记：

1. LiDAR 点云结合 SLAM 位姿，铺成机器人周围的局部栅格。
2. 每格用 RANSAC 找地面平面，以点面残差 RMSE 预估前方粗糙度。
3. IMU 去掉约 27.5 Hz 的机械峰值，按速度归一化并在 100 个样本窗口内计算真实振动。
4. RLS 用遗忘因子 0.995 学习几何分数到车体响应的比例与偏置，实时修正振动成本图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03720v1)
- [arXiv](https://arxiv.org/abs/2609.03720v1)

---

<a id='2609.03657v1'></a>
## [Rethinking 3D Noise: Learning 3D-Aware Video Priors via Optimization-Free Morphological Perturbations](https://arxiv.org/abs/2609.03657v1)

**Authors:** Onat Şahin, Mohammad Altillawi, George Eskandar, Carlos Carbone, Ziyuan Liu

**Published:** 2026-09-03

**Categories:** cs.CV

**Abstract:**

3D scene representations like NeRF and 3D Gaussian Splatting (3DGS) suffer severe artifacts in sparse-view settings. Recent generative 3D artifact fixers attempt to address this, but rely on paired corrupted and clean renders requiring costly, per-scene reconstructions across varying view configurations. While 2D image augmentations act as instant regularizers, no explicit equivalents exist for 3D representations to preserve spatial consistency across views, an essential property for 3D-aware training. We propose 3D Morphological Perturbations as an optimization-free regularizer that preserves spatial consistency. Leveraging explicit 3DGS, we treat each Gaussian as a fundamental building block - analogous to a 2D pixel - and apply perturbations across its morphological parameter space via scale, rotation, and pruning. Our method eliminates per-scene 3DGS optimization loops from dataset curation while enabling models to learn stronger geometric priors than sparse-view baselines in diagnostic ablations conducted on a lightweight video diffusion sandbox. Scaled to a 14B-parameter video model via ControlNet, our approach maintains visual fidelity while reducing mean depth error by 12.5% over state-of-the-art image-to-image 3D artifact refiners, ultimately boosting downstream robotics policy success rates by up to 8.0% across 3 of 4 manipulation tasks.

### 论文解读

#### 摘要翻译

论文题为“重新思考三维噪声：通过免优化的形态学扰动学习三维感知视频先验”。作者关注稀疏视角三维重建中常见的表面缺失、深度错误和跨视角不一致。现有生成式修复方法往往需要对每个场景重新优化三维高斯泼溅（3DGS），再制作退化渲染与清晰渲染的配对样本，因而数据构造昂贵、难以扩展。

论文提出三维形态学扰动（3D Morphological Perturbations，3DMP）：直接修改3DGS的三维原语，再沿同一相机轨迹渲染视频。这样得到的条件视频在所有帧中共享同一套三维退化，视频模型可以借助时间与视角证据恢复几何，而不是逐帧凭纹理猜测。论文的主张是，这种无需逐场景优化的退化方式能够学习可迁移的三维感知视频先验，并改善深度和机器人操作表现。

#### 研究问题与动机

驱动力是“图像看起来合理”与“空间结构真正一致”之间的差距。稀疏相机观测使3DGS产生空洞、漂浮物和不正确的表面；若对每一帧单独使用模糊、抖动或噪声增强，增强结果之间没有共同的三维原因，模型容易生成时间闪烁或多视角冲突的细节。即使对整个图像序列做统一变换，也无法自然模拟某些高斯原语缺失后随相机位置变化的遮挡和投影变化。

现有逐场景优化配对数据虽然较真实，却把训练成本绑定到每个场景的重建过程。论文的核心假设是：在渲染之前扰动共享的三维表示，能让退化视频保留跨视角一致的结构线索；其中随机删点所造成的表面缺失，尤其接近稀疏重建的关键错误。模型若要恢复这些缺失，就必须综合邻近帧、可见边界和场景上下文。

#### 核心方法

整体流程是：读取或构建可渲染的3DGS，向高斯的位置、辐射属性和形态参数施加3DMP，沿相机轨迹渲染退化视频，以清晰视频作为监督训练条件扩散模型，最后生成修复视频并进行颜色校正。随机扰动发生在原语层，因而同一个三维改变会同时影响多个投影，而不是把每帧当作互不相关的图像。

位置采用 μ'i=μi+εμ，噪声按场景高斯尺度的平均值标定，γxyz=0.002。颜色和不透明度分别加入标准差0.05的高斯噪声；颜色裁剪到合法范围，不透明度裁剪到[10^-4,1-10^-4]。形态部分以pprune=0.5的Bernoulli掩码随机删去高斯，近似制造一半密度的缺失表面；尺度噪声标准差为σscale=0.01并设置10^-6下界，旋转四元数加入σrot=0.01的噪声后重新归一化，约对应1.15度的角度变化。删点负责模拟几何空洞，尺度与旋转扰动负责模拟局部形状不准，辐射扰动则覆盖颜色和透明度误差。

轻量沙盒使用Stable Diffusion v1.5与AnimateDiff v1.5.2，将4通道噪声潜变量和4通道条件潜变量拼成8通道输入，并在空间、时间注意力中加入rank=64、alpha=128的LoRA。大规模系统使用14B参数的Wan2.2 DiT和膨胀式ControlNet，复制8个DiT block，膨胀步幅为3；新增条件通道用原权重的0.1倍初始化，以避免训练初期条件信号过强。Wan训练视频为832×480、每段49帧，采用AdamW、余弦学习率、5% warmup和bfloat16，在8张80GB GPU上训练30个epoch。推理用Flow-Match Euler调度器进行50步去噪，再在CIELAB空间匹配全局颜色统计。理论上，一阶近似将渲染扰动写作V(η)≈V0+JRη；在扰动样本上训练会抑制预测器对非结构化变化的敏感性，形成与渲染雅可比有关的流形正则。CKA分析中3DMP的中间激活漂移较小，支持其训练稳定性的解释。

#### 方法对比与创新

与2D Render Blur或逐帧抖动相比，3DMP的本质差异是先改三维原语、后渲染序列，所以退化天然共享几何原因。与需要为每个场景优化稀疏3DGS的修复器相比，它不依赖逐场景优化轨迹，能按统一规则批量生成退化—清晰训练对。与把位置、颜色、透明度和形态噪声不加区分地全部叠加的naive方案相比，论文的消融显示有针对性的形态退化更有效，退化强度并非越大越好。

创新贡献包括把“数据退化设计”从像素空间移到显式三维表示空间、用随机剪枝逼近真实表面缺失，以及将这一退化接入从LoRA沙盒到Wan+ControlNet的不同规模视频生成器。它适合稀疏视角新视图补全、深度修复和重视空间稳定性的机器人感知；若任务只要求单张图像的审美修补，三维一致性带来的额外代价未必必要。

#### 实验结果

诊断实验在DL3DV-10K的250个场景上进行，200个训练、50个测试；极端S80设置每80个视角取一帧。单独使用3DMP达到PSNR 14.907、SSIM 0.517、F-score 0.762，优于3D朴素扰动的F-score 0.710和2D渲染模糊的0.724，说明几何形态缺失是有效训练信号。

在DL3DV六视角、无视角重叠的高难度设置中，Wan+ControlNet（Morph.）深度RMSE为29.53、AbsRel为0.306、δ1为0.576；Fixer为54.24、0.555、0.424，Difix为29.87、0.308、0.553。中等难度下RMSE也由Difix的31.67降至30.65。下游RLBench的DREMA评测中，Close Jar成功率从51.2%升至59.2%，Pick Cup从34.4%升至38.0%，Lift从23.6%升至24.4%；Insert Peg仍只有约1.6%，表明视觉改善不能消除精密控制瓶颈。

#### 实用指南

复现时首先需要可渲染的3DGS及对应相机轨迹，再按论文给出的删点概率、尺度、旋转和辐射噪声生成条件视频；训练对由DL3DV-10K、ScanNet、ScanNet++按75%、15%、10%混合构成，共约1万对。应固定832×480、49帧、50步Flow-Match Euler推理等设置，并检查颜色校正是否改变几何评测。论文给出了模型结构和主要超参数，但没有详细报告Wan文本提示模板、每个场景的高斯数量分布或单视频推理秒数，复现这些部分时不能擅自补写。

实现检查可以按三个层次进行：先确认同一高斯掩码被用于整段序列，再确认删点后仍能正常进行可见性排序和渲染，最后分别比较单类扰动与组合扰动。训练数据最好保留清晰目标和扰动条件的配对关系，并在训练、测试场景之间严格隔离；评测时除了像素指标，还应记录深度尺度处理、视角重叠规则和机器人控制器是否改变。这样才能判断提升来自三维先验，而不是来自数据泄漏、颜色后处理或控制策略差异。

论文正文未承诺一个可直接安装的完整开源实现，因此工程迁移的重点是接口而非复制某个仓库：在已有显式三维原语上实现随机剪枝和参数扰动，沿真实或合成相机轨迹渲染，再把条件序列接到视频扩散模型的ControlNet或同类适配器。迁移到不同数据集时，应重新检查场景尺度、相机覆盖和稀疏程度；若使用NeRF等隐式表示，需先设计等价的结构扰动，不能直接照搬高斯参数操作。

#### 局限性与意义

3DMP依赖显式、可渲染的三维原语和足够的时空上下文，对动态物体、严重遮挡或本身不可靠的3DGS，收益可能下降。平面区域如桌面偶尔会产生新的噪声表面；删点概率与各类噪声幅度存在非线性耦合，目前主要依靠固定超参数，缺少针对场景可见性自动调节的策略。Insert Peg约1.6%的成功率也提醒我们，视觉修复不能替代标定、碰撞建模和高精度运动控制。

方法的意义在于提供了一个简洁的三维一致性归纳偏置：不必为每个场景反复优化，就能从共享原语制造成对视频，并把结构恢复压力传给生成模型。它把评价链条从画面质量延伸到深度误差和操作成功率，不过这些收益并非所有任务都稳定出现。实际部署仍应同时测量跨视角一致性、深度、时延与控制闭环表现，而不能只依据视觉指标判断可用性。

#### 总结

核心思想：在三维高斯上制造缺失，让视频模型学会结构补全。

1. 输入可渲染的3DGS和相机轨迹。
2. 对高斯随机删点，并扰动位置、尺度、旋转、颜色和透明度。
3. 沿共享轨迹渲染退化视频，与清晰视频配对训练条件扩散模型。
4. 用50步采样和颜色校正生成修复结果，再检查深度与机器人任务。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03657v1)
- [arXiv](https://arxiv.org/abs/2609.03657v1)

---

<a id='2609.03729v1'></a>
## [Unfold The World: Factorize 4D Properties in Reinforcing Spatial Reasoning](https://arxiv.org/abs/2609.03729v1)

**Authors:** Yijun Yang, Shenghe Zheng, Wenbo Li, Jianhui Liu, Haoze Sun, Yanbing Zhang, Jiaxiu Jiang, Lin Song, Haoyang Huang, Nan Duan, Lei Zhu

**Published:** 2026-09-03

**Categories:** cs.CV

**Abstract:**

Despite the remarkable prowess of Vision-Language Models (VLMs) in general multimodal tasks, they remain fundamentally ``flat'' when reasoning about the physical world. We argue that this spatial bottleneck stems from a profound dimensional mismatch: while VLMs are trained to interpret 2D projections, true spatial reasoning demands the recovery of latent 3D geometry and temporal continuity. To conquer this high-dimensional complexity, we advocate a shift from monolithic learning to a ``divide and conquer'' paradigm. We present FactoSR, a factorized reinforcement learning framework that explicitly interpret the dimensions collapsed by visual projection. At its core, FactoSR decomposes the monolithic problem of world-consistent reasoning into three orthogonal, geometric sub-objectives: planar correspondence ($XY$), depth consistency ($Z$), and temporal reversibility ($T$). By optimizing these verifiable constraints within a unified policy learning mechanism, we effectively transform an ill-posed projection recovery problem into a series of tangible reasoning steps. Extensive evaluations on multi-view and video benchmarks demonstrate that this elegant decomposition yields substantial gains in 3D and 4D reasoning, achieving a 5.9% boost on VSI-Bench and 4.5% on All-Angles-Bench. Our findings suggest that reinforcing explicit, factorized 4D consistency is a critical step toward evolving VLMs into robust, world-aware reasoners.

### 论文解读

#### 摘要翻译

视觉语言模型在通用多模态任务上进展显著，但由于主要从二维投影学习，仍难以稳定理解三维几何、跨视角关系和时间连续性。论文提出 FactoSR，一种分解式空间强化学习框架，将复杂的四维推理拆分为平面对应（XY）、深度一致性（Z）和时间可逆性（T）三个可验证目标，并结合语义准确率进行优化。该方法旨在让模型从依赖二维外观捷径，转向遵守世界几何和时序约束的推理。

#### 研究问题与动机

核心问题是：怎样在不为每一种空间任务设计独立模型的情况下，使 VLM 形成连贯的空间世界表征？现有 SFT 扩容和空间 token 往往只增加表面模式；统一优化 4D 目标又难以定义稳定的监督，模型因而会把物体大小、纹理等外观线索误当作深度，多视角定位和动态路线尤其脆弱。作者的驱动力是把“空间正确”拆为相互补充的物理事实，并让每个事实都能被程序化检查。

论文的基本假设是，XY、Z、T 是复杂空间能力中相对正交的组成部分：对应关系应满足相机变换后的重投影，深度至少应满足正确的前后秩序，动作和路线应在时间反向时保持可逆。这样的因子化奖励可以提供比单一答案正确率更有指向性的学习信号。

#### 核心方法

FactoSR 以 Qwen3-VL-8B-Instruct 为骨干，采用两阶段流程。第一阶段是空间感知 SFT，使用约 8.2M 样本，并采用短答案到长答案/思维链的课程：约 90.3% 为短答案，9.7% 为长答案或 CoT。数据结合通用视觉理解和 1.2M 级空间专项样本，覆盖定位、对应关系等任务。其跨帧推理模板 Anchor–Transfer–Verify 要求模型先在源帧找到可靠锚点，再把锚点传到目标帧，最后核验转移是否成立。

第二阶段使用 VeRL 中的 GRPO。模型针对每个问题生成 8 个候选回答，以组内相对表现估计优势，因此不需额外价值网络；KL 正则则抑制策略过度偏离参考模型。奖励首先检查格式，格式正确后再计算加权和：语义准确率 R_acc、XY 对应奖励 R_XY、Z 深度奖励 R_Z 和 T 时间奖励 R_T。

这种“先格式、后内容”的门控很重要：模型必须把坐标、关系或解释写成解析器可识别的结构，几何验证器才能可靠工作。GRPO 的组内比较还可以在同一问题的多个候选中偏好物理一致的推理，而不必为每个连续几何量构造人工价值标签。因子权重 λ1–λ4 因而承担任务侧的取舍：语义答案保证任务完成，XY、Z、T 分别约束不同维度的世界一致性。

XY 奖励利用相机内参 K、位姿 T 和深度 d，将源视图像素的射线反投影到三维，再变换并投影到目标视图。可见性 mask 会过滤目标视图中被遮挡或不可见的点，避免把不可观测性误惩罚为错误。Z 奖励不要求模型输出难以稳定获得的绝对深度，而使用 Kendall-τ 衡量物体前后顺序；顺序越一致，奖励越高。T 奖励检查时间循环一致性，例如正向“向右转”在逆向轨迹中应恢复为“向左转”。论文给出的 SFT 全局 batch 为 128、序列长度 8192，基础与 CoT 学习率分别为 5×10^-5 和 1×10^-5；RL 学习率为 1×10^-6，采样温度为 1.0。

#### 方法对比与创新

与只把空间问答当作普通语言监督的方法相比，FactoSR 将模型回答放入相机几何和时序闭环中验证；与直接预测绝对三维坐标相比，Z 因子先学习更稳健的相对深度；与 vanilla GRPO 相比，它不是只放大“答对”的样本，而是区分错误究竟来自平面对应、深度顺序还是时间方向。主要创新包括因子化奖励体系、Anchor–Transfer–Verify 推理组织方式，以及把可见性和相对秩序纳入奖励计算。

从训练哲学看，方法不是增加一个独立的深度网络或显式 3D token，而是保留普通 VLM 的语言推理接口，用外部可计算的约束筛选其生成轨迹。这样既能复用大规模视觉语言预训练，又能把“看起来合理”的回答与真正满足投影方程的回答区分开。对比实验中 T 因子在路线规划上的大幅增益尤其说明，时间反向约束能补足静态图像监督无法表达的方向性。

其中，XY 约束主要回答“是不是同一个空间点”，Z 约束回答“谁更靠前”，T 约束回答“沿时间倒放是否仍自洽”；三者分别处理身份、层次与方向。把这些问题拆开后，错误分析也更直接，研究者可以判断模型究竟缺少视觉匹配、空间排序还是运动理解能力。

这也使后续诊断和扩展更容易。

这套设计适合多视角理解、视频空间关系、机器人导航和路线规划等具有相机标定或时序结构的场景。若任务缺少可靠位姿，XY 奖励需要改为学习式或弱监督几何检查；若任务关心精确测距，则还需在 Z 因子上补充尺度监督。

#### 实验结果

实验覆盖 VSI-Bench、All-Angles-Bench 以及通用 3D/4D 和多模态基准，并与 GPT-4o、Gemini-2.5-Pro、InternVL3、Qwen2.5-VL、SpatialThinker、VST 等比较。FactoSR-8B-RL 在 VSI-Bench 获得 61.5%，比基础模型高 5.9 个百分点；在 All-Angles-Bench 获得 55.4%，提升 4.5 个百分点，通用 3D/4D 平均约 62.0%。

消融进一步证明奖励设计的作用：仅用 vanilla GRPO 的提升只有 0.2%；加入因子后，XY 对应、Z 深度和 T 路线任务分别带来约 2.7、1.3 和 7.9 的增益。模型在 MMBench、MMStar 等通用能力测试上仍具竞争力，表明空间强化没有明显损害通用视觉语言能力。

#### 实用指南

论文公开了 FactoSR 实现（GitHub 仓库为 ZimaBlue-WAM/FactoSR），并基于 Qwen3-VL 与 VeRL 构建。复现时应先准备多视角相机内参、位姿、深度及可见性信息，再生成 Anchor–Transfer–Verify 格式的 SFT 轨迹；训练时遵守两阶段学习率、8192 序列长度和 RL 每题 8 次 rollout 的设定，并确保格式奖励在几何奖励之前生效。RL 数据约 32K 条，其中 81.2% 是空间样本、18.8% 是数学样本。

迁移到导航时，可把 XY 换成路标/目标点重投影，把 Z 换成障碍物或路标的相对深度，把 T 换成往返路线一致性；迁移到视频时则需要可靠的帧间相机运动或替代性的轨迹约束。实际部署还需检查推理阶段能否获得标定信息，以及几何验证的计算开销。

工程上还应分别记录四类奖励及其方差，而不是只看总 reward；否则某一因子可能因尺度过大掩盖其他能力。数据预处理要统一像素坐标、齐次坐标和相机坐标系的约定，并对不可见点显式标记。若使用不同骨干或更长视频，可保留同样的奖励接口，先冻结通用视觉编码器做小规模验证，再逐步扩大 rollout，以降低错误几何标注导致的训练不稳定风险。

#### 局限性与意义

该框架依赖训练阶段提供的相机内参、位姿和深度相关信息，标定误差会直接污染 XY 奖励；复杂动态遮挡、不可见区域和非刚体运动也会削弱几何检查。Kendall-τ 只保证相对深度顺序，不等于恢复真实尺度；此外，因子权重和奖励之间的平衡仍需针对任务调整。

尽管如此，论文的重要意义在于展示了一条把 VLM 从二维相关性推向可验证世界规律的路径：不必一次性学习难以定义的完整 4D 表征，而是用多个互补约束逐步塑造一致推理。该思想也为机器人感知与语言规划之间的接口提供了较清晰的训练范式。

#### 总结

核心思想：用可验证因子强化四维空间推理。

速记 pipeline：
1. 用多视角和跨帧样本训练模型寻找空间锚点。
2. 把锚点转移到新视图并检查对应关系。
3. 用相机几何检查平面位置、用秩相关检查前后深度。
4. 用正逆时间轨迹检查动作可逆性，再以 GRPO 联合优化。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03729v1)
- [arXiv](https://arxiv.org/abs/2609.03729v1)

---

