time: 20260916

# Arxiv Computer Vision Papers - 2026-09-16

## Table of Contents

1. [World Models for Embodied Intelligence: From Plausible to Controllable to Actionable](#2609.16697v1)
2. [Artificial Intelligence-Enabled Space Robot Operations: Technologies, Challenges and Prospects](#2609.16880v1)
3. [UniDex-ViTac: Learning Unified Visuo-Tactile Dexterous Manipulation Policy from Human Video Data](#2609.16504v1)
4. [The Robot Data Factory](#2609.16705v1)
5. [EgoPathBench: Evaluating Zero-Shot Egocentric Waypoint Decision-Making in Vision-Language Models](#2609.16610v1)

---

## Papers

<a id='2609.16697v1'></a>
## [World Models for Embodied Intelligence: From Plausible to Controllable to Actionable](https://arxiv.org/abs/2609.16697v1)

**Authors:** Nanjie Yao, Hao Wang, Chong Cheng, Zhikang Chen, Wenzhe Li, Jiafei Lyu, Li Shen, Peilin Zhao, Zongqing Lu, Gao Huang, Steven Hoi, Dacheng Tao, Deheng Ye

**Published:** 2026-09-15

**Categories:** cs.RO, cs.AI

**Abstract:**

World models connect perception and decision-making in embodied intelligence by maintaining hidden state, anticipating consequences, comparing interventions, and adapting when execution departs from expectations. Although progress is often measured by visual fidelity, their value lies in improving behavior. Before reaching for a cup, a person anticipates its weight and resistance to grasping, shaping the hand before contact. Such anticipation is coarse and rarely pictorial, yet it guides action. This raises a central question: which predictive capabilities improve behavior? Existing surveys, organized by architecture, output modality, or application domain, leave this question implicit. We introduce three progressively stronger capability levels: Plausible models preserve task-relevant temporal, geometric, or physical structure; Controllable models additionally predict how interventions alter that structure; and Actionable models translate predictions into measurable gains in planning, action, learning, evaluation, verification, recovery, or data selection. We complement this hierarchy with a 3 x 4 matrix crossing geometry, physics, and action grounding with improvement loops centered on data, rewards, policies, and the model itself. Using this framework, we survey manipulation, navigation, locomotion, autonomous driving, and general embodied learning, tracing technical progressions, clarifying capability requirements, and examining datasets, benchmarks, and evaluation protocols. We identify challenges in long-horizon consistency, uncertainty calibration, causal intervention testing, latency, verification and recovery, and cross-embodiment transfer. This perspective shifts evaluation from visual plausibility toward whether predictions capture task-relevant state, reflect intervention effects, and improve the closed-loop behavior of embodied agents.

### 论文解读
#### 摘要翻译
本文综述具身智能中的世界模型，并提出以决策为中心的能力分类：Plausible（貌似真实）要求预测保持任务相关的几何、物理结构；Controllable（可控）要求正确预测动作干预的后果；Actionable（可行动）要求预测切实改善规划、学习、评估或部署。
#### 方法动机分析
本文旨在解决一个核心问题：现有方法的痛点是许多视觉模型能生成逼真画面，却不一定知道物体会如何运动。具身智能面对部分可观测、接触丰富和长时域任务，需要推断未观测状态、预测动作后果并及时纠错。因此论文主张以行为效用而非单纯视觉质量衡量世界模型。长时程状态漂移、相关性冒充因果性、跨机器人动作语义不一致和推理成本，是从“貌似真实”走向“可行动”的主要障碍。
#### 方法设计详解
作者将问题抽象为：输入历史观测、历史动作和指令，形成状态摘要后，预测未来状态与奖励信号 p(z未来,r未来|历史,a未来)。状态 z 可是像素、潜变量、物体 slot、BEV 场景或物理参数。候选动作序列经过模型得到未来轨迹，再用于三类闭环：搜索或 MPC 选择动作；在潜空间想象中训练策略；执行时比较预测和真实观测以检测故障。论文还构建 3×4 矩阵：几何、物理、动作三种 grounding，分别与数据、奖励、策略、模型自改进四种 loop 组合。这样既检查“预测是否合理”，也检查“干预是否忠实”及“是否带来决策增益”。
#### 方法对比分析
不同于按网络架构或应用场景整理，本文按预测对行为的贡献分层。视觉逼真度只是必要条件；更大的 backbone 或更长的预测时域不会自动带来更强的控制能力。矩阵还能区分数据生成、奖励评估、策略想象和部署后自修复等机制，适合分析操控、驾驶与导航系统的瓶颈。
#### 实验分析（精简版）
本文是综述，没有针对单一新模型的统一实验结果或新的量化 benchmark，因此不能报告“本文提升了多少”。论文给出的定量组织证据是 3×4 矩阵：3 种 grounding（几何、物理、动作）与 4 种 improvement loop（数据、奖励、策略、模型自改进）交叉。作者建议分层评估：视觉层使用 SSIM、LPIPS、FVD，几何层关注深度/占据误差和漂移率，可控性关注动作预测与反事实排序，可行动性关注成功率、planning regret、数据效率及故障检测 precision/recall。文中覆盖 RoboNet、BridgeData V2、RLBench、CALVIN、RH20T、nuScenes、Waymo、CARLA、Gibson 和 Matterport3D。
#### 实用指南
复现具体方法时，应依据对应原论文确认数据划分、预处理、batch、学习率、训练轮数、硬件和代码；本文没有统一超参数或一个统一开源实现。迁移到新机器人时，通常需重新处理动作空间、相机/几何标定和接触动力学，并验证长时程漂移及干预一致性。已有项目是否完全开源也应逐项核对其原始项目页与许可证。
#### 总结
核心思想：决策效用定义世界模型价值。
1. 历史观测与动作形成状态摘要；
2. 预测几何、物理和动作后果；
3. 检验干预忠实度并接入搜索、想象学习或运行期验证；
4. 用真实任务收益判断是否可行动。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.16697v1)
- [arXiv](https://arxiv.org/abs/2609.16697v1)

---

<a id='2609.16880v1'></a>
## [Artificial Intelligence-Enabled Space Robot Operations: Technologies, Challenges and Prospects](https://arxiv.org/abs/2609.16880v1)

**Authors:** Zeyuan Huang, Gang Chen, Zixuan Hao, Guoqin Tang, Junyi Zong, Guoyou Ban, Jiale Wang, Haoyang Lv, Chaoqian Ren, Sitong Liu

**Published:** 2026-09-15

**Categories:** cs.RO

**Abstract:**

Space robots are increasingly expected to perform long-duration, contact-rich, and multi-stage operations with limited human intervention. Recent advances in artificial intelligence (AI), robot learning, and embodied foundation models provide new opportunities to improve the autonomy and adaptability of such systems, but their transfer to space is constrained by scarce mission data, space-specific dynamics and sensing conditions, limited onboard resources, and stringent safety requirements. This article reviews artificial intelligence-enabled space robot operations (AI-SRO) from a capability-building perspective. We first summarize representative operational scenarios, autonomy trends, and space-specific constraints. We then establish a three-layer technical framework comprising capability foundations, capability formation, and capability deployment/evolution. Within this framework, we review simulation environments, datasets and benchmarks; task and environment understanding, state perception, decision-making and planning, and action execution; and onboard deployment, ground-to-space adaptation, continual learning, and capability transfer. Finally, we propose key research directions toward trustworthy simulation and data, open-world multimodal cognition, long-horizon safe decision-making, physically constrained policy learning, and space computing infrastructures.

### 论文解读
#### 摘要翻译
本文综述人工智能赋能空间机器人操作（AI-SRO）的进展。面对空间站、在轨服务和深空探测中的长时域、多接触、多阶段任务，作者提出“能力构建”视角，将研究归纳为能力基础、能力形成、能力部署与演化三层，并总结仿真、数据、感知决策、动作执行、在轨部署和持续学习。

#### 方法动机分析
本文的驱动力来自空间站、在轨服务和深空探测任务对长时域自主操作的需求。传统空间机器人依赖精确模型、预设程序和地面遥控，难以应对非合作目标、极端光照、接触不确定性、通信延迟与少人工干预。学习方法更灵活，却可能破坏动力学约束，且真实异常数据和在轨算力有限。论文因此强调：自主能力必须同时满足适应性、物理一致性、可部署性与安全性。

#### 方法设计详解
这是系统综述，不是单一新模型。第一层“能力基础”从任务需求出发，构建同时包含微重力、自由浮动基座—机械臂耦合、接触动力学和极端视觉条件的仿真环境，并融合真实、合成及跨域增强数据，以成功率、碰撞能量和泛化能力评估。第二层“能力形成”接收语言指令、视觉/力觉观测和机器人状态：先由大语言模型分解任务并完成指令接地，再用视觉、三维重建、位姿估计及视触觉融合得到状态，接着进行长时域任务规划和约束运动规划，最后通过强化学习、模仿学习或视觉—语言—动作模型输出控制。视触觉融合可在遮挡和微弱接触力下联合视觉与力矩推断目标状态；控制奖励还应考虑动量补偿和基座扰动。第三层使用领域随机化、课程学习缓解仿真到真实偏差，并用剪枝、量化、模型压缩和知识蒸馏适配低功耗抗辐射硬件，再利用在轨经验持续学习和跨任务迁移。训练示例依赖Isaac Sim、MuJoCo和并行GPU，论文未规定统一学习率或延迟。

#### 方法对比分析
与只研究感知、规划或控制的工作相比，该框架把仿真数据、能力学习和部署演化连为闭环；与传统模型驱动控制相比，AI更适合非结构化和非合作场景，但必须增加物理约束与安全监控。ROS-Gazebo/Space ROS适合系统验证，Isaac Sim/GRADE偏高逼真视觉，SPART/Basilisk偏动力学建模，工具选择应服务于对应证据。

#### 实验分析（精简版）
论文统计2020—2026年52项核心研究：动作学习与控制约15项（约28.8%），能力基础11项（约21.2%），理解感知和部署演化约各9项（约17.3%），显示控制最活跃而基础与部署仍不足。文章讨论Astrobee、SPEED/SPEED+、SpaceDet等基准，但没有统一方法的准确率、成功率或消融结果，因此结论主要支持研究版图，不代表某方案已普遍优于传统方法。异常数据稀缺、物理违规、算力鸿沟和形式化安全不足是主要限制。

#### 实用指南
可用ROS-Gazebo或Space ROS搭系统，用Isaac Sim/MuJoCo训练，并以SPART或Basilisk检查动力学；参考Astrobee Dataset、SPEED/SPEED+和SpaceDet。应随机化光照、材质、动力学参数和噪声，分别评估成功率、碰撞能量、跨域泛化与实时性。文中提及Space ROS、Astrobee、SPART及SPEED+可公开获取，但未说明本文自身统一代码或模型。迁移时需替换动力学、传感器标定和安全约束并重训。

#### 总结
核心思想：能力闭环实现可信空间自主操作。
1. 建立高可信仿真与多源数据基础。
2. 将语言指令接地为感知、规划和控制。
3. 以视触觉和物理约束增强接触鲁棒性。
4. 通过域适应、轻量部署和持续学习演化。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.16880v1)
- [arXiv](https://arxiv.org/abs/2609.16880v1)

---

<a id='2609.16504v1'></a>
## [UniDex-ViTac: Learning Unified Visuo-Tactile Dexterous Manipulation Policy from Human Video Data](https://arxiv.org/abs/2609.16504v1)

**Authors:** Hyesung Lee, Si-Hwan Heo, Sungwook Yang

**Published:** 2026-09-15

**Categories:** cs.RO

**Abstract:**

Human videos provide demonstrations of dexterous manipulation but lack robot-executable actions and tactile measurements. We present UniDex-ViTac, a framework that uses human-video-guided simulation to generate robot demonstrations paired with fingertip contact observations for training a deployable visuo-tactile policy. Object-specific residual reinforcement learning specialists adapt annotated human-object interaction references to a robotic arm-hand system. Their successful rollouts pair final robot action targets with robot-side fingertip contact observations. From 50 human demonstrations across ten objects, we collect 10,000 simulated trajectories to train a single Action Chunking with Transformers (ACT) based generalist. The policy combines point clouds, proprioception, and four binary contact signals encoded through fingertip labels and a separate token, without requiring human references or privileged object identity and pose at deployment. The contact-augmented configuration achieves 68.3% macro-average success in simulation, compared with 55.5% for the point-cloud-only baseline. Without real-robot demonstrations or policy fine-tuning, it succeeds in 73/110 physical trials (66.4%) across six seen and five unseen objects, compared with 60/110 (54.5%) for the baseline, an increase of 11.8 percentage points. These results support the feasibility of learning a unified visuo-tactile dexterous manipulation policy from video-guided simulated interactions. Project page: https://unidex-vitac.github.io/

### 论文解读
#### 摘要翻译
UniDex-ViTac 旨在仅利用人类视频数据，为多指机器人手学习可部署的视觉—触觉灵巧操作策略。由于视频没有机器人动作和触觉，作者在仿真中训练残差强化学习专家，把人类示范适配到机器人系统，并生成 10,000 条模拟轨迹，再训练单一 ACT 通用策略。真实世界 110 次试验成功 73 次，成功率 66.4%，比仅视觉方法高 11.8 个百分点。
#### 方法动机分析
灵巧手遥操作成本高，而人类视频便于扩展，却存在形态差异和接触信息缺失。论文假设人手—物体运动仍包含可迁移的任务结构，残差 RL 能在保留参考动作的同时修正机器人运动学与接触动力学；少量指尖接触信号则可弥补视觉遮挡。当前验证范围是单一抓取并抬升技能。
#### 方法设计详解
首先从 DexYCB 重建手、腕部、指尖和物体参考。随后每个物体训练一个 residual RL specialist，输出围绕参考轨迹的增量控制，并通过限幅积分避免偏离过大；奖励同时考虑物体、指尖、腕部跟踪、接触激活和运动惩罚。成功轨迹汇总后训练 ACT generalist：PointNet++编码 512×6 点云，结合本体感知与四位接触位作为输入；接触既作为独立 token，又作为点云中的指尖空间标记，输出期望腕部位姿和相对指尖目标。仿真以接触力超过 1.0 N 置位，硬件使用气压传感器。训练 1,500 epochs，推理 20 Hz、每次预测 30 步动作，并用时间集成平滑控制。
#### 方法对比分析
与依赖真实动作或特权物体位姿的 ACT、Diffusion Policy、BC-Transformer 不同，本方法把特权信息限制在仿真专家阶段，最终策略使用可部署感知。与单一多任务 RL 相比，先按物体解决适配再蒸馏，减少联合探索难度；与 PCD Only 相比，二值触觉提供接触条件，而非单纯增加视觉容量。
#### 实验分析（精简版）
仿真专家成功率 94.3%，通用策略为 68.3%。真实实验中 PCD Only 为 54.5%，加入接触后的 UniDex-ViTac 为 66.4%，即 110 次中成功 73 次；未见物体上为 62.0% 对 54.0%。接触融合消融显示 token-only 56.4%、one-hot-only 56.8%，双路径 68.3%，说明空间接触标记与全局接触模式互补。局限是四位信号缺少力大小，且仿真与硬件接触区域存在差异，结果主要支持抓取并抬升场景。
#### 实用指南
数据使用 DexYCB，仿真使用 Isaac Lab；项目页已提供，但正文未明确完整代码开源状态。复现需保持 8,192 并行仿真环境、每物体约 1,000 条轨迹、512×6 点云、四位接触接口及 20 Hz/30 步动作块。迁移到其他机器人时需重做运动学残差适配、接触阈值校准和策略训练。
#### 总结
核心思想：残差专家把人类视频变成触觉示范。
速记 pipeline：
1. 提取人手—物体参考并重建交互轨迹；
2. 用物体专属残差 RL 适配机器人并筛选成功轨迹；
3. 用 ACT 融合点云、本体状态和接触位，输出平滑动作块。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.16504v1)
- [arXiv](https://arxiv.org/abs/2609.16504v1)

---

<a id='2609.16705v1'></a>
## [The Robot Data Factory](https://arxiv.org/abs/2609.16705v1)

**Authors:** Sami Haddadin, Ivan Laptev, Ian Reid, Dezhen Song, Cesare Stefanini, Abdalla Swikir, Xingxing Zuo, Lyes Saad Saoud, Mahmoud Hamandi, Mohamed Heshmat, Oualid Doukhi, Abdeldjallil Naceri, Attique Bashar, Abdelrahim Mohamed, Teodor Tomic, Yue Peng, Samuel Schneider, Cheng-Chung Lee, Janine Guo, Qinghao Zhang, Kim Jeffery

**Published:** 2026-09-15

**Categories:** cs.RO

**Abstract:**

Physical AI requires more than increasingly large robot datasets: intelligent robots acquire knowledge through continuous interaction with the physical world. We argue that the defining scientific resource of Physical AI is therefore not raw robot data alone, but robot experience - physically grounded interaction whose observations, actions, embodiment, context, and outcomes preserve the perception-action-consequence loop. We introduce the Robot Data Factory (RDF), a mission-driven infrastructure and methodology for continuously generating, validating, benchmarking, and reusing such experience. RDF organizes heterogeneous robots and environment-specific training grounds through reproducible missions, skill curricula, synchronized multimodal sensing, external ground truth, an agentic robot network, data pipelines, and living benchmarks. Rather than treating datasets as static end products, RDF implements a closed Deploy-Measure-Learn-Repeat cycle in which validated physical experience supports world models, vision-language-action models, embodied policies, digital twins, and subsequent robot deployment. We further formalize robot experience and its quality, introduce a mission-task-skill-episode-dataset-benchmark-capability hierarchy, and derive quantitative scaling laws and an algorithmic synthesis procedure connecting robot fleet size, sensor rates, storage, learning representations, tokenization, training compute, inference, and latency to Embodied-AI cluster requirements. The framework is instantiated in three complementary physical training grounds for domestic, environmental, and energy applications. RDF thus reframes robot data generation as a continuous scientific production process and provides a pathway toward reproducible, scalable, and eventually federated infrastructure for Physical AI.

### 论文解读

#### 摘要翻译
论文提出 Robot Data Factory（RDF），一种以使命为驱动的训练场框架，用于持续生成、验证和工程化“机器人经验”。它不是静态数据集，而是由环境专属训练场、机器人网络、数据管线和排行榜组成的持续基础设施，服务于 Physical AI 的规模化发展。

#### 方法动机分析
驱动力在于：视觉语言模型依赖数字语料，却不能替代机器人在物理世界中主动交互。现有机器人数据集的痛点是绑定单一平台和任务、缺少外部真值，也难保留感知、动作与后果的闭环；精细操作还受到触觉、身体形态和环境动力学共同影响。RDF 的核心假设是，可验证、可复现、带具身上下文的“经验”应被视为一等科研资源。

#### 方法设计详解
RDF 将流程组织为 Mission→Task→Skill→Episode→Dataset→Benchmark→Capability。机器人在 Home、Environment、Energy 等 chamber 中执行使命，控制方式可为学习策略、遥操作或经典控制；系统通过 Deploy→Measure→Learn→Repeat 闭环迭代。Ubiquitous Data Pipeline 覆盖物理、平台、传输、可观测性和管理层，并用 PTP 保证多传感器同步，记录机器人、物体、环境和其他智能体的耦合观测。经验保留学习元组 (o_t,a_t,o_{t+1},m_t,y_t)，其中包含阶段元数据与评价标签。Agentic Robot Network 将机器人、传感器和仪表盘作为网络节点；Pitstop Leaderboard 则分别评价人类、遥操作、自主策略和世界模型预测。使命分数同时受难度、安全门控、数据有效性门控及技能加权分数影响，使“完成任务”与“产生可信数据”绑定。论文还给出按舰队规模、传感率、数据复用和训练周期估算算力的模型。

#### 方法对比分析
RDF 的本质不是提出一个新网络，而是把采集、验证、评分和再部署整合为可持续的数据生产系统。相比单平台模仿学习数据，它保留跨具身和多模态经验；相比一次性 benchmark，它用排行榜形成持续反馈，并显式检查安全与数据质量。该框架适合建设多机器人、长期运行的 Physical AI 实验设施，但跨机构采用需要统一协议与治理。

#### 实验分析（精简版）
论文主要提供基础设施规模分析，而非策略精度实验：假设 120 台机器人时，单参考机器人保留数据约 38.41 MB/s，舰队摄入约 6.50 GB/s，30 天数据约 11.95 PB；训练 7B 模型的月度舰队语料约需 2,566 个加速器。优化数据选择和复用后，示例需求可降至约 170 个。上述数字支持资源规划，但尚不能证明 RDF 必然提升任务成功率。

#### 实用指南
项目网站为 agentic-robotics-lab.github.io/robot-data-factory。论文计划开放使命与训练场规格、具身元数据、验证规则、评分脚本和 benchmark 协议；完整代码、模型和数据集当前未被明确说明为已发布。复现时应优先实现时间同步、多模态记录、学习元组、质量门控和安全评分。迁移到其他机器人需重新标定传感器，替换具身元数据、技能定义和难度权重，并重新采集验证数据。

#### 总结
核心思想：把机器人经验做成生产系统。
1. 使命定义可复现实验目标。
2. 多机器人在互补训练场执行任务。
3. 同步管线记录带标签闭环经验。
4. 门控与排行榜筛选可信数据。
5. 改进模型再部署，循环产生更难经验。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.16705v1)
- [arXiv](https://arxiv.org/abs/2609.16705v1)

---

<a id='2609.16610v1'></a>
## [EgoPathBench: Evaluating Zero-Shot Egocentric Waypoint Decision-Making in Vision-Language Models](https://arxiv.org/abs/2609.16610v1)

**Authors:** Yang Zhao, Zhuo Chen, Xubo Yang

**Published:** 2026-09-15

**Categories:** cs.CV

**Abstract:**

Zero-shot waypoint navigation requires vision-language models to select, from the current first-person observation, a sequence of spatial actions that is feasible for the agent and reaches the goal, placing joint demands on the integrated spatial intelligence of today's foundation VLMs. Existing spatial-intelligence benchmarks primarily evaluate isolated judgments of relations, directions, or targets and therefore do not directly measure the integrated navigation ability required to combine target recognition, action-consequence assessment, distance estimation, and path planning. To fill this evaluation gap, we introduce EgoPathBench, a dataset and five-task benchmark for first-person waypoint decision-making. Each question presents an egocentric RGB image, a natural-language goal, and numbered visible waypoints; a model returns traversable candidates or an ordered route. Predictions are evaluated for candidate feasibility, adjacent-edge legality, and goal arrival under point-agent or embodied geometry. EgoPathBench contains 31,852 training, 1,345 validation, and 1,111 benchmark questions and retains at least one geometrically verified reference route for every route question. Across nine VLMs, the highest EgoPath Score is only 28.3. The top-ranked model reaches 35.9% success on Point Path, but only 2.9% and 4.0% on Embodied Path and Intent Path, respectively, showing that current models remain limited in forming complete, goal-consistent routes under embodiment constraints. Beyond the evaluation data, we release the corresponding training resource. Fine-tuning Qwen 3.5 4B on the released training split raises its EgoPath Score from 3.9 to 38.9 and improves all four reported evaluations across three external spatial benchmarks, with gains of 1.4--9.6 points.

### 论文解读
#### 摘要翻译
EgoPathBench 是评估视觉语言模型（VLM）零样本第一人称航点决策的基准。它不只问模型能否识别方向、距离或物体关系，还要求模型结合目标识别、几何约束与路径规划，输出可实际通行的路线。基准包含点代理可通行性、具身可通行性、显式点路径、显式具身路径和意图具身路径五类任务。

#### 方法动机分析
传统空间基准的痛点是偏重孤立判断，无法说明模型能否连续行动。真实机器人还必须考虑自身占据空间、障碍物和目标接地，因此论文用直径 0.6 m 的具身代理检验净空，并用 3D 场景几何验证路线。核心问题是：VLM 的语义知识能否转化为物理可行的第一人称决策？当前设定聚焦单视角，未覆盖动态环境中的长期闭环导航。

#### 方法设计详解
数据来自 InternScenes 的室内 3D 资产（源自 3RScan、ScanNet、ARKitScenes 和 Matterport3D）。作者把地面候选点及墙壁、家具表面等负样本投影到图像，再依据 3D 几何分别建立点代理和具身可通行图，并为每题生成经验证的参考路径。输入是第一人称图像、候选点和目标名称或意图描述；模型先判断可通行点，或从意图中定位目标，再输出有序航点。评测用 Balanced Accuracy/F1 衡量点判断，用 Valid Path Rate、Success Rate 和 SPL 衡量路径，EgoPath Score 为五项任务指标的等权平均。数据含 31,852 条训练、1,345 条验证和 1,111 道测试题。微调实验冻结视觉塔，以 LoRA rank=8、学习率 1e-4 在两张 A100 上训练 Qwen 3.5 4B。

#### 方法对比分析
相比只测空间关系的基准，EgoPathBench 的关键区别是把“看懂场景”与“沿合法边持续走到目标”连接起来；相比只看终点的导航评估，它还检查每条连续路径边及身体碰撞约束。显式目标任务测试规划，意图任务进一步测试目标接地，适合诊断 VLM 到具身智能之间的落差。

#### 实验分析（精简版）
九个主流 VLM 的零样本表现显示，最佳 Gemini 3.1 Pro 的 EgoPath Score 仅 28.3。点路径成功率可达 35.9%，加入身体约束后具身路径降至 2.9%，意图路径为 4.0%；约 96% 的模型虽能给出合法第一步，却难以保持后续路径合法。LoRA 微调使 Qwen 3.5 4B 得分由 3.9 升至 38.9，并令三个外部空间基准提升 1.4–9.6 个百分点。局限是单视角和标准化场景，不能直接等同于真实机器人闭环性能。

#### 实用指南
复现需保留候选点投影、3D 碰撞验证、0.6 m 身体约束及逐边路径检查，不能只比较终点；论文未说明 epoch、batch size 和推理温度。作者承诺开源训练数据、题集、评估代码和模型配置，但发布前不应假定资源已可下载。迁移到其他机器人时要重设身体尺寸、相机视场并重建可行性真值；动态场景还需补充时序观测与记忆。

#### 总结
核心思想：用几何真值测VLM走路
速记：
1. 投影正负航点并用几何验证；
2. 从目标名或意图规划具身航点；
3. 以合法率、成功率和 SPL 评测；
4. 用空间 CoT 数据微调并检验迁移。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.16610v1)
- [arXiv](https://arxiv.org/abs/2609.16610v1)

---

