time: 20260915

# Arxiv Computer Vision Papers - 2026-09-15

## Table of Contents

1. [From Prediction to Decision: World-Model-Guided Action Selection for Continuous Pile Excavation](#2609.15382v1)
2. [WLA$^3$: World Latent Action Modeling for Semantics, Dynamics, and Kinematics](#2609.15870v1)
3. [JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion](#2609.15770v1)
4. [Goal-Oriented Communications for Physical AI: Design and Testbed](#2609.15895v1)
5. [GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving](#2609.15169v1)
6. [SURE-Map: Self-Correcting Streaming Geometric Foundation Model](#2609.15795v1)
7. [Unsupervised Point Cloud Registration via Training-Time Semantic Guidance](#2609.15228v1)
8. [LieSpline-DP: Lie-Group B-Spline Diffusion Policy for Smooth Robot Manipulation](#2609.15162v1)
9. [X-WBC: A Cross-Embodiment Foundation Model for Humanoid Whole-Body Control](#2609.15213v1)
10. [Touch2Trace: Tactile-Driven Imitation Learning for Dexterous Cable Tracing](#2609.15921v1)

---

## Papers

<a id='2609.15382v1'></a>
## [From Prediction to Decision: World-Model-Guided Action Selection for Continuous Pile Excavation](https://arxiv.org/abs/2609.15382v1)

**Authors:** Ailing Zhang, Fan Gao, Song Zhang, Kawa Leong, Ziyu Wu, Yafei Wang

**Published:** 2026-09-14

**Categories:** cs.RO

**Abstract:**

Wheel-loader excavation is a sequential decision problem in which every scoop changes the terrain available to subsequent actions. A practical world model must predict action consequences accurately, rank candidates in real time, and operate inside the closed loop of a full-size machine. We present the World-Action Model (WAM), which proposes multiple scoops, rejects geometrically inadmissible candidates, jointly predicts signed terrain change and loaded volume, executes the candidate with the largest predicted load, and replans from the newly observed terrain. On 32 geometry-disjoint MinSlope test episodes, adding world-model ranking to matched diffusion proposals reduces the mean scoop count from 651.8 to 540.6 (17.1%), preserves 32/32 completion, and improves every paired episode. In a complete-system comparison, WAM completes 32/32 episodes versus 29/32 for an independently trained soft actor-critic policy. Comparisons of input representations, spatial support, and five architectures identify an accurate and efficient physics-structured predictor. We further evaluate the interface on event-disjoint full-size-loader data and deploy the complete perception-proposal-prediction-selection-execution loop for autonomous excavation. The ROS2/TensorRT implementation processes five candidates in 72.4 ms on a Jetson AGX Orin. The simulation results establish decision-level gains, while the physical experiments demonstrate real-world closed-loop feasibility.

### 论文解读

#### 摘要翻译

论文把轮式装载机挖掘视为连续序列决策：每次铲取都会改变后续地形。作者提出世界—动作模型（WAM），先生成多个动作，过滤几何非法候选，再预测地形变化和装载量，执行预测装载量最高者并重新规划。在32个几何不相交的MinSlope回合中，WAM将平均铲取次数从651.8降至540.6，减少17.1%，完成率为32/32；Jetson AGX Orin上五候选推理耗时72.4 ms。

#### 方法动机分析

堆料会被持续耗尽，当前动作既决定眼前装载量，也改变下一次可挖区域。启发式难覆盖复杂约束，SAC等无模型策略又难显式检查动作后果。WAM的假设是：用动作条件世界模型预测候选的短期物理结果，就能在实时预算内改善选择。它聚焦单步重排，而非完整的多步长期规划。

#### 方法设计详解

输入是10 m×10 m、0.2 m分辨率的50×50高度图和表面法向量；每个动作由朝向、横向偏移、接近距离、挖掘长度、最大深度、速度比例、铲斗翻转角组成。条件扩散器以地形和目标填充率生成16个提案；几何过滤器检查工作空间、碰撞、容量/扫掠体积并保持候选差异，留下5个。预测器接收地形及扫掠掩码、挖掘深度图，输出有符号地形变化和装载体积。其编码器—解码器基于约10.48M参数的DigNet++ Large，采用多尺度残差块和空洞卷积；地形变化区域约5倍加权，体积分支用Smooth L1，总损失为L_H+L_V。选择预测装载量最大者执行，再用新地形闭环；推理部署在Jetson AGX Orin，五候选延迟为72.4 ms。

#### 方法对比分析

Diffusion-direct生成后直接执行一个动作，WAM的创新贡献是增加“多候选—后果预测—重排”层，把生成与决策分开；SAC直接由策略输出动作，WAM则显式预测变化并允许执行前硬过滤。相较几何启发式，WAM能用学习到的物理结构适应多样地形。它适合已有地形观测、动作后果数据和实时算力的装载机系统，但对域变化仍需重新训练或校准。

#### 实验分析（精简版）

在32个MinSlope测试回合中，WAM为540.6次、4.78 m³、32/32完成；匹配的Diffusion-direct为651.8次、4.03 m³、32/32，SAC为596.2次且29/32完成。加入动作扫掠掩码和挖掘深度图后，地形预测MAE降低43.2%，装载量预测MAE降低51.3%。跨仿真器与真实数据的接口评估IoU均至少0.641；真机实验主要证明闭环可行，长期规划和大规模实物效率对比仍是局限。

#### 实用指南

论文说明了ROS2/TensorRT和Jetson AGX Orin部署，但未给出可核实的公开代码或模型链接。复现需保持50×50地形表示，生成法向量及两类动作图，训练扩散提案与世界模型，并实现16提案、几何过滤、5候选排序；明确设定包括λ_V=1和变化区域约5倍加权。迁移到新材料、机器或仿真器需重新采集动作—地形数据并校准几何、容量和坐标约束，不能视为零样本迁移。

#### 总结

核心思想：世界模型让挖掘先预测再决策。

速记：
1. 生成多样的七维挖掘动作。
2. 用几何规则过滤并保留可行候选。
3. 预测地形变化与装载量后排序。
4. 执行最高预测动作并重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15382v1)
- [arXiv](https://arxiv.org/abs/2609.15382v1)

---

<a id='2609.15870v1'></a>
## [WLA$^3$: World Latent Action Modeling for Semantics, Dynamics, and Kinematics](https://arxiv.org/abs/2609.15870v1)

**Authors:** Peidong Liu, Zhiyuan Xiang, Mingyang Li, Wenhao Li, Jiale Zhang, Jiahao Sun, Jiawei Li

**Published:** 2026-09-14

**Categories:** cs.RO

**Abstract:**

Scaling generalist policy models with heterogeneous data is limited by the lack of unified, low-noise action supervision. Human egocentric videos are abundant, but only a small fraction comes with high-quality hand-action labels. Observed world transitions offer a common source of action-related supervision across data sources. We introduce WLA$^3$ (World Latent Action Modeling for Semantics, Dynamics, and Kinematics), a unified generalist policy model framework built around representations learned by a World Latent Action Model (WLAM). WLAM first learns how multimodal world states change over a local interval, encoding synchronized camera views and available embodiment-state changes into a compact local latent action and a richer transition feature. Reconstruction from partial modalities and consistency across overlapping windows encourage robust transition representations. WLA$^3$ reuses them across semantics, dynamics, and kinematics: local latent actions support action-sensitive physical-dynamics modeling, segment-level features directly supervise the VLM through a Semantic Latent Aggregate (SLA), and an action expert jointly predicts latent actions together with embodiment-specific robot controls. Human videos provide scalable transition supervision, while robot trajectories ground the shared representation in executable native controls. On LARYBench, the final 32D latent action reaches 67.89\% average classification accuracy. WLA$^3$ achieves 81.9% average success across six real-robot tasks versus 66.2% for $π_{0.5}$. Performance improves as generalist policy model mid-training data scales, and human videos support human-to-robot transfer. Project page can be found at https://wla-3.github.io/.

### 论文解读
#### 摘要翻译
WLA³（World Latent Action Modeling for Semantics, Dynamics, and Kinematics）提出统一机器人策略框架。其世界潜在动作模型 WLAM 从人类视频、机器人轨迹、仿真与 UMI 数据学习跨具身表示，把世界状态变化编码为紧凑潜在动作和变换特征，并用于语义、动力学与运动学。六项真实机器人任务平均成功率达到 81.9%。

#### 方法动机分析
人类视频规模大但缺少精确动作标签，机器人轨迹有本体状态和控制监督却数量有限、动作空间绑定硬件。单视角 LAM 还容易受遮挡和投影歧义影响，且通常只服务预训练。WLA³ 假设不同具身的状态变化可以共享潜在动作，多视角和本体状态提升可辨识性，再由机器人专属模块恢复可执行控制。

#### 方法设计详解
输入为同步主视角/手部视角图像、机器人状态及本体配置。Transition Encoder 编码状态对，得到 1024 维变换特征；Variational Bottleneck 将其压缩为 32 维连续潜在动作。Decoder 根据初始状态和潜在动作重构终态的图像、状态等模态，目标结合像素、LPIPS、光流、DINOv2 语义和深度损失，并用重叠时间窗的一致性约束稳定动作方向。SLA 对约 30 步策略段的变换特征池化，监督 VLM 学习物理语义；LAC-WM 以观察和潜在动作为条件，通过 Conditional Flow Matching 预测未来视觉特征；LARA 用 Perceiver 联合预测共享潜在动作与特定机器人的原生动作。策略推理因此形成“理解变化—预测后果—输出控制”的链路。论文使用 84.1K 小时混合数据；完整 batch、学习率和训练轮数未说明。

#### 方法对比分析
相较 UniVLA、LAPA，WLA³ 增加多视角、本体状态和段级 SLA；相较 DreamDojo，它用潜在动作条件化世界模型，区分不同交互带来的后果。贡献在于把共享状态变化表示同时接入语义、预测和控制，而不只是把 LAM 当预训练器。它适合视觉操作及人类视频到机器人的知识迁移。

#### 实验分析（精简版）
LARYBench 语义分类平均准确率为 70.75%，高于 DiLA 的 63.26%；控制回归 MSE 为 0.37，优于 CLAP 的 0.47 和 DreamDojo 的 0.51。AgiBot G1 六项真实任务平均成功率 81.9%，比 π0.5 的 66.2% 高 15.7 个百分点。消融显示加入 LAC-WM、LARA、SLA 后成功率由 72.1% 提升至 76.3%、79.8% 和 81.9%。局限是对视觉/本体感知依赖较强，触觉与力反馈覆盖不足。

#### 实用指南
项目主页为 wla-3.github.io；论文未明确说明代码、权重和数据是否全部公开。复现需准备同步多视角数据、机器人状态和本体配置，保留重叠窗口及约 30 步段聚合，并实现多模态重构损失。换用新机器人时可复用共享表示，但需重训 LARA 原生动作头，并统一相机、控制频率、动作尺度和状态定义。

#### 总结
核心思想：潜在动作贯通三层策略。

速记：
1. 多视角状态对编码为变换特征与潜在动作。
2. 解码终态并用重叠窗口约束动作一致。
3. SLA 提取物理语义，LAC-WM 预测动作后果。
4. LARA 将共享表示翻译为机器人控制。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15870v1)
- [arXiv](https://arxiv.org/abs/2609.15870v1)

---

<a id='2609.15770v1'></a>
## [JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion](https://arxiv.org/abs/2609.15770v1)

**Authors:** Qihao Yuan, Yixuan Qiu, Ziyu Cao, Ming Cao, Kailai Li

**Published:** 2026-09-14

**Categories:** cs.RO

**Abstract:**

Light detection and ranging (LiDAR) remains less explored than RGB-D sensing for perceptive legged locomotion, and existing LiDAR-based approaches often rely on explicit mapping. We present JEPLO (Joint-Embedding Predictive learning for legged LOcomotion), a single-stage learning framework for mapping-free, LiDAR-based perceptive locomotion for legged robots. We introduce a proprio-exteroceptive JEPA (PE-JEPA) world model to learn predictive egocentric terrain representations from onboard observations, including raw LiDAR scans. A concurrent JEPA-teacher-student (CJTS) pipeline is further proposed to train a locomotion policy informed by JEPA latent representations in simulation using deep reinforcement learning with a simple reward formulation. The framework achieves successful sim-to-real transfer, enabling omnidirectional traversal of diverse terrains, including long staircases and high boxes, with lightweight onboard computation. Evaluations demonstrate greater robustness than existing perceptive locomotion frameworks, particularly under degraded perception caused by occlusion, sparsity and noise. Further analysis validates JEPLO's ability to retain task-relevant information under these challenging conditions. We open-source our implementation, experimental datasets, and hardware setup designs https://github.com/ASIG-X/JEPLO.

### 论文解读
#### 摘要翻译
JEPLO 是一种面向 LiDAR 腿足运动的单阶段、无地图学习框架。它用本体感知—外感知联合嵌入预测架构 PE-JEPA 学习自我中心地形表征，并以并发 JEPA 教师—学生（CJTS）训练策略。在仿真和 Unitree Go2 实机上，系统可通过长楼梯、高箱体，并抵抗遮挡、稀疏扫描、噪声及黑暗环境。

#### 方法动机分析
RGB-D 易受光照影响，而许多 LiDAR 方法依赖 SLAM、里程计和显式地图，带来计算负担及场景变化敏感性。单帧 Livox 扫描又稀疏且不规则。作者的核心假设是，预测未来本体与外感知的潜码，比重建全部几何细节更能保留与运动有关的信息；同时，教师必须考虑学生实际可见性受限的问题。

#### 方法设计详解
输入包括 10 帧本体历史、2 帧深度图和 5 帧动作。MLP 将本体信息编码为 32 维，ViT 将深度图编码为 64 维，GRU 保存时序状态；预测头依据状态和动作预测下一时刻两类潜码，以平方误差对齐目标编码器输出。SIGReg 用 Epps–Pulley 统计量约束潜码接近各向同性高斯，防止塌缩，总损失中的权重 λ=0.1。五帧原始 LiDAR 扫描累积为 60×25 深度图。特权教师读取地形高度图和状态，学生只读取 JEPA 潜码与本体历史；PPO 训练策略，并用 MSE 对齐师生嵌入。JEPA 以 10 Hz 更新，控制以 50 Hz 运行。

训练在 IsaacLab 中完成，随后用 MuJoCo 做跨仿真验证；推理时学生不再访问特权高度图，而以 JEPA 的时序潜码和本体历史驱动控制器，这正是部署可行性的关键。

#### 方法对比分析
PIE 侧重高度图重建，WMP 使用 DreamerV3 式观测重建；JEPLO 的区别是直接在嵌入空间预测未来，并联合本体和外感知信息。PE-JEPA 是主要新机制，SIGReg、ViT、GRU 和 PPO 是配套组件；CJTS 则在策略训练中并发协调教师与受限学生。它更适合无地图、算力有限且传感器会退化的腿足机器人。

#### 实验分析（精简版）
在 0.2 m 楼梯和 0.5 m 箱体仿真任务中，JEPLO 的箱体成功率为 100%，楼梯成功率为 59.2%；Stripe 遮挡下楼梯成功率为 62.6%，退化小于 PIE、WMP。No-Prop 和 No-Occ 消融分别说明本体信息及遮挡增强不可缺少。Go2 实机成功攀爬 53 cm 箱体和 7.5 m 室外楼梯，并在黑暗、窗口胶带遮挡和单帧扫描下行走。Jetson AGX Orin 上 CPU/GPU 利用率为 12.4%/7.35%，内存 1.84 GB；高速运动下的失败边界仍缺乏系统评估。

#### 实用指南
论文提供 https://github.com/ASIG-X/JEPLO，并声称包含训练、部署、MuJoCo 验证、数据集和硬件设计。复现需保持五帧累积、60×25 投影、10/50 Hz 频率、历史长度和 λ=0.1，并加入 LiDAR 外参漂移、遮挡、摩擦和电机参数随机化。训练在 IsaacLab 中进行、MuJoCo 验证，RTX 6000 Pro 约需 12 小时。换机器人时要重设状态/动作接口、LiDAR 外参与动力学，并重新训练。

#### 总结
联合潜码预测驱动鲁棒运动

1. 编码本体与 LiDAR 历史。
2. 结合动作预测未来潜码并正则化。
3. 并发蒸馏特权教师到可部署学生。
4. 在传感器退化随机化下训练并迁移实机。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15770v1)
- [arXiv](https://arxiv.org/abs/2609.15770v1)

---

<a id='2609.15895v1'></a>
## [Goal-Oriented Communications for Physical AI: Design and Testbed](https://arxiv.org/abs/2609.15895v1)

**Authors:** Shutong Chen, Wenkai Zhang, Adnan Aijaz, Miao Guo, Yansha Deng

**Published:** 2026-09-14

**Categories:** cs.RO, eess.IV

**Abstract:**

Physical AI relies on frequently-updated, latency-sensitive video stream to perceive, reason, and interact with the physical world, resulting in strict latency requirements with much higher data volumes that existing 5G networks cannot support. Goal-oriented communication (GoC) offers as a promising approach to solve this challenge by transmitting only task-relevant semantic representations. However, existing GoC frameworks were mainly evaluated in the simulations while their effectiveness has never been validated in a practical deployment of physical AI application. In this work, we develop an end-to-end GoC testbed for Physical AI, which connects a PiPER robot arm equipped with an RGB-D camera and a 5G modem to an NVIDIA Jetson AGX Orin edge server through a 5G OpenAirInterface network. We propose and implement three GoC frameworks that transmit 3D bounding boxes, 2D scene graphs, and 3D scene graphs, as three types of semantic representations, respectively. They share the common functional modules designed for closed-loop Physical AI applications, including semantic extraction, full stack 5G transmission, language model inference, digital twin validation, and robotic control. Extensive experiments on our testbed show that our GoC frameworks reduce the task completion time by up to 52.6% and improve task success probability by up to 45%, compared to the traditional framework that periodically transmits the raw image data. These results validate the practical effectiveness of our GoC framework and pave the way for efficient and reliable Physical AI applications over future 6G networks. Project website: https://sites.google.com/view/goc-physical-ai-testbed.

### 论文解读
#### 摘要翻译
物理人工智能依赖高频、低时延视觉流，但高清图像造成巨大带宽压力。论文提出目标导向通信（GoC），仅传输三维边界框或二维/三维场景图等任务相关语义，并搭建连接 PiPER 机械臂、RGB-D 相机、真实 5G 网络和边缘服务器的端到端测试平台。相较原始图像传输，任务完成时间最多缩短52.6%，成功率提高45个百分点。

#### 方法动机分析
机器人闭环通常需要1–10 Hz控制；30 fps Full-HD视频的数据率可达1.44 Gbps。以往研究多在仿真中忽略端侧提取、真实协议栈和网络缓冲。GoC的假设是，抓取和规划真正需要的是对象身份、位置及关系，而不是全部像素，因此可用符号语义替代图像；代价是可能损失纹理和形状信息。

#### 方法设计详解
机器人用RGB-D相机获得图像与深度。检测对象并将深度反投影后，生成含标签、三维中心和几何量的3D-BBox；也可构造对象关系三元组，形成2D场景图，或结合点云得到含前后关系的3D场景图。紧凑文本经5G上行至边缘，LLM/针对任务微调的SLM把符号输入转为结构化pick等动作。动作先在MuJoCo数字孪生中进行碰撞检查和RRT-Connect路径规划，再经下行发送给PiPER执行。总时延由语义提取、上下行、语言模型、数字孪生和执行组成；新增提取开销由通信量和视觉推理量下降抵消。论文提到Llama-3.2-1B等SLM，但未说明完整训练超参数。

#### 方法对比分析
传统方案上传PNG并让边缘VLM直接看图；GoC将轻量感知前移，传输任务相关事实，并在执行前增加数字孪生安全校验。它不是单纯图像压缩，而是重定义通信接口：3D-BBox服务几何规划，场景图服务关系推理，3D-SG补充深度方向。该方法适合对象和空间关系清晰的操作任务，对布料等形变物体则可能不如原始视觉。

#### 实验分析（精简版）
在水果装箱、积木堆叠测试中，语义消息约0.038–0.78 KB，PNG约150 KB，数据量减少99.48%–99.97%。GoC最高缩短52.6%的完成时间；3D-SG加数字孪生校验时，成功率由传统框架的40%达到85%。结果支持其带宽与闭环效率优势，但实验任务范围有限，端侧检测成本和语义丢失仍需评估。

#### 实用指南
复现需要RGB-D标定、对象检测与深度反投影、OAI 5G、边缘LLM/SLM、MuJoCo和RRT-Connect，并统一网络、推理和执行计时。论文提供项目网站，但未明确完整代码、权重和数据集许可，开源状态应另行核验。迁移时需替换坐标系、类别、动作格式、数字孪生模型及碰撞约束，并重新训练或验证模型。

#### 总结
核心思想：只传任务所需的空间语义

1. RGB-D感知对象与深度；
2. 构造边界框或场景图；
3. 5G传语义，边缘模型生成动作；
4. 数字孪生验证后回传执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15895v1)
- [arXiv](https://arxiv.org/abs/2609.15895v1)

---

<a id='2609.15169v1'></a>
## [GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving](https://arxiv.org/abs/2609.15169v1)

**Authors:** Xiao Liu, Haoyu Li, Jianghao Leng, Lin Wang, Chao Sun

**Published:** 2026-09-14

**Categories:** cs.CV, cs.RO

**Abstract:**

Driving vision-language-action (VLA) models increasingly reason before acting, but their intermediate reasoning is often weakly grounded in physical scene evidence and loosely connected to executable behavior. We present GRAVA, a framework built around Grounded Reasoning-to-Action (GRA), which unifies grounding, reasoning, and action generation in a single autoregressive stream. GRA links action-relevant language references to 2D visual regions and ego-centric physical states, organizes object interactions and decisions in a trajectory-anchored typed graph, and serializes this structure into grounded reasoning. A single VLM generates this reasoning followed by a compact Executable Planner action that is deterministically decoded into a continuous trajectory. We further introduce an agentic GRA data construction pipeline that combines forward scene grounding with backward trajectory anchoring, and use it to build GR-NavSim with 2.2M grounded question-answer pairs and 70K GRA reasoning traces. A progressive training strategy develops grounded cognition through pre-training, establishes the reasoning-to-action interface through imitation, and improves driving behavior through reinforcement learning and exploration. Using about 60% of the available human driving demonstrations for action supervision, GRAVA-8B achieves state-of-the-art performance among purely autoregressive driving models on the full NAVSIM benchmark. On an internal long-tail benchmark, full GRA improves key-object compliance and Closed-loop Driving Score by 19.3% and 20.5% over action-only prediction, respectively. These results show the benefit of preserving action-relevant physical evidence from grounded reasoning through executable action generation.

### 论文解读

#### 摘要翻译

GRAVA 面向自动驾驶视觉-语言-动作模型，解决语言推理缺少物理接地、推理难以传递到动作的问题。它把对象、交互和决策组织成轨迹锚定类型图，让模型先生成带视觉引用和物理量的推理，再生成可执行运动原语，并通过四阶段训练共同优化推理与规划。

#### 方法动机分析

传统模型说“前方车辆需避让”，却未必指出是哪辆车、距离多远、速度如何；推理与规划分开学习时，正确理由也可能没有转化成安全动作。GRAVA 假设每个决策都应能回溯到接地对象，并最终连接到动作锚点，从而把解释变成规划所需的中间状态。它重点面向复杂交互和长尾场景，跨帧一致性仍有限。

#### 方法设计详解

输入是视觉观测、自车状态和导航指令。GRA 类型图包含场景上下文、接地对象、交互、对象级决策、自车决策与动作锚点；推理中的 bounding-box、距离和速度等信息必须服务于后续动作。VLM 自回归生成推理 r，再生成 a=(p,g,φ)：p 为 STOP、CRAWL、CURVE、CRUISE 等运动原语，g 为挡位，φ 为参数。策略写作 π(r|x)π(a|x,r)，固定几何解码器据此输出连续轨迹。训练依次使用 2.2M 接地 QA 做预训练、规划器 SFT、筛选高奖励样本的自蒸馏，以及只在“有改进潜力”的可恢复场景上进行主动 RL；奖励作用于完整推理—动作序列。

#### 方法对比分析

相比 DriveLM/DriveVLM，GRAVA 显式绑定框和物理状态，并输出结构化原语参数而非仅自然语言或航点。相比 UniAD/PARA-Drive，它保持自回归 VLM 形式，不依赖额外 3D 感知头。类型图和动作锚定增强可追溯性，确定性解码增强执行可控性；代价是原语集合可能限制更复杂控制。

#### 实验分析（精简版）

在 NAVSIM 全协议中，GRAVA-8B 达到 90.48 PDMS。长尾场景中，相比仅预测动作的版本，关键对象合规性提升 19.3%，闭环驾驶分数提升 20.5%；把低质量推理替换为高质量接地推理后，胜率从 3% 升至 55%，说明接地推理确实影响动作质量。主要边界是单帧接地、时序一致性和推理开销。

#### 实用指南

论文提供代码地址 https://github.com/AhernResearch/grava。复现需准备 nuPlan/NAVSIM 数据以及 GR-NavSim 的接地 QA 和 GRA 轨迹，保留对象引用、物理量与原语参数格式，并同时评估 PDMS、KOC 和 CDS。迁移到新车辆或新数据集时，需要替换对象/状态标注、运动原语和几何解码器并重新训练接口；论文未明确所有依赖版本、学习率和硬件细节。部署时还应检查轨迹解码后的动力学可行性与闭环安全约束。

#### 总结

核心思想：接地推理驱动可执行动作。

1. 类型图锚定对象、交互与决策。
2. 生成带框和物理量的推理。
3. 生成原语参数并几何解码轨迹。
4. 用轨迹奖励联合优化推理与动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15169v1)
- [arXiv](https://arxiv.org/abs/2609.15169v1)

---

<a id='2609.15795v1'></a>
## [SURE-Map: Self-Correcting Streaming Geometric Foundation Model](https://arxiv.org/abs/2609.15795v1)

**Authors:** Mingkai Liu, Hao Zhao, Xingxing Zuo

**Published:** 2026-09-14

**Categories:** cs.CV

**Abstract:**

Streaming geometric foundation models are emerging as a compelling alternative to SLAM systems. Yet this streaming nature introduces a fundamental issue: each prediction is made from limited context, which is vulnerable to dynamic objects and weak textures. Small local errors accumulate into severe geometric distortion and long-horizon scale drift. We argue that reliable streaming reconstruction requires geometric foundation models to be not only predictive, but also self-correcting. We introduce SURE-Map, a self-correcting framework built upon two complementary principles. First, we explicitly model cross-view geometric uncertainty. Unlike conventional depth or point confidence, which primarily reflects the reliability of individual-view prediction, our uncertainty directly measures whether the jointly predicted pose and depth induce geometrically consistent cross-view pixel correspondences. Second, because local correction alone cannot eliminate slowly accumulating scale errors, we introduce multi-timescale self-correction: fast consecutive-frame inference preserves streaming efficiency, while sparse keyframe-window inference provides longer-range geometric evidence to periodically recalibrate the scale of recent trajectories. SURE-Map establishes new state-of-the-art performance for online feed-forward reconstruction across long-horizon benchmarks, reducing ATE-RMSE from 24.00 to 17.24 m on KITTI, 5.11 to 4.74 m on Oxford Spires, and 31.37 to 28.58 m on VBR, with further improvements to 15.17, 4.63, and 22.12 m when incorporating loop-closure refinement. Project page: https://mingkai-liu.github.io/projects/sure-map/.

### 论文解读
#### 摘要翻译

SURE-Map 面向流式几何基础模型的误差累积问题，联合建模跨视图几何不确定度与多时间尺度自纠错：连续帧保持低延迟处理，稀疏关键帧周期性校准长期尺度，从而改善轨迹和三维重建。

#### 方法动机分析

离线几何模型可利用长上下文，流式模型却只能依赖局部因果观测。动态物体、弱纹理和退化视角造成的微小深度/位姿误差会持续传递，最终形成几何畸变和尺度漂移。作者的关键假设是，深度与姿态共同诱导的跨视图对应关系，比单视图深度置信度更能揭示联合几何是否可靠；稀疏关键帧则可提供长期尺度约束。

#### 方法设计详解

连续 RGB 帧先输入 LingBot-Map 主干，得到几何 token、深度和相对位姿。系统用深度与位姿把当前像素投影到上一帧，得到诱导光流；不确定度头读取几何特征，预测光流对数方差，并以真实光流残差构造高斯 NLL 训练。推理时，高不确定度区域会被过滤，点到平面优化也按不确定度加权，通过 Gauss–Newton 修正局部平移。与此同时，快路径逐帧运行，慢路径周期性对稀疏关键帧窗口做全注意力推理；通过逆深度拟合得到尺度系数，再把尺度校准平移与局部优化平移融合。主干冻结，仅在 TartanAir 上训练不确定度头 20k iterations；额外单帧开销约 15–30 ms。

#### 方法对比分析

相较 COLMAP、DROID-SLAM、DPVO，SURE-Map 保留前馈几何基础模型的形式，同时引入显式的可靠性估计和修正。相较离线长上下文方法，它不需访问完整序列；相较其他流式前馈方法，它把跨视图一致性用于点过滤、加权优化和尺度重校准。适用对象是在线建图、移动机器人和连续视频，但全注意力关键帧处理会带来算力开销，且仍受骨干几何质量限制。

#### 实验分析（精简版）

在 KITTI、Oxford Spires、VBR 上用 ATE-RMSE 测轨迹，在 Neural RGB-D、7-Scenes 上测重建，并与 COLMAP、DROID-SLAM、DPVO、离线几何模型及其他流式方法比较。KITTI 中 LingBot-Map 的 ATE 为 24.00 m，SURE-Map 降至 17.24 m，加入回环后进一步到 15.17 m；Oxford Spires 由 5.11 m 降至 4.74 m，VBR 由 31.37 m 降至 28.58 m。Neural RGB-D 的 F1 达 66.20%。消融表明尺度再校准是抑制长期漂移的主因，KITTI ATE 可由 24.00 降至 17.57，不确定度加权补充局部修正。代价是 KITTI 吞吐从 11.68 FPS 降到 9.17 FPS，因此实时性与精度仍需权衡。

#### 实用指南

论文提供项目页面（mingkai-liu.github.io/projects/sure-map/），但材料未明确说明代码、模型和数据是否全部开源，应以页面链接核实。复现需实现诱导光流、NLL 不确定度头、加权 Gauss–Newton、关键帧全注意力和逆深度尺度拟合，并保留流式状态。迁移时应重新校验标定、尺度和监督，通常需要重训不确定度头，并调整关键帧间隔与过滤阈值。

#### 总结

核心思想：流式几何预测的自查与多尺度纠错

1. 主干预测连续帧几何。
2. 跨视图光流残差学习不确定度。
3. 不确定度过滤并加权局部优化。
4. 关键帧全局对齐、周期校准尺度。
5. 融合快慢路径输出稳定地图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15795v1)
- [arXiv](https://arxiv.org/abs/2609.15795v1)

---

<a id='2609.15228v1'></a>
## [Unsupervised Point Cloud Registration via Training-Time Semantic Guidance](https://arxiv.org/abs/2609.15228v1)

**Authors:** Kezheng Xiong, Shiyun Xu, Sheng Ao, Siqi Shen, Cheng Wang, Chenglu Wen

**Published:** 2026-09-14

**Categories:** cs.CV

**Abstract:**

Unsupervised registration of large-scale LiDAR point clouds remains challenging due to the geometric ambiguity inherent in outdoor scenes, which degrades pseudo-label quality and leads to suboptimal convergence, particularly for sparse, low-resolution scans such as those from nuScenes. We reveal that registration models intrinsically encode semantic awareness that strongly correlates with registration accuracy, albeit without explicit semantic supervision. However, this native awareness is fragile: noisy supervision arising from geometric ambiguity in unsupervised settings rapidly erodes the learned semantic structure, causing performance collapse. To this end, we propose CAESAR, a teacher-student framework guided by an off-the-shelf 3D segmentation model exclusively during training. We observe that potential inlier matches are often buried just beneath a few spurious neighbors in the noisy feature space, motivating Dual-Cue Guided Re-Matching to recover them through reselection rather than simply rejecting. Building on this, a train-only Semantic-Geometric Label Mining performs lightweight, batch-specific teacher refinement and mines reliable pseudo-labels under semantic guidance. We further introduce Semantic Predictive Distillation to consolidate the student's semantic awareness in the feature space. Extensive experiments on KITTI and nuScenes demonstrate state-of-the-art performance, with pronounced gains on the challenging nuScenes benchmark. Crucially, CAESAR incurs zero inference overhead and requires no semantic annotations on the registration data. Code will be released.

### 论文解读
#### 摘要翻译
论文面向大规模 LiDAR 点云的无监督配准。稀疏点云存在几何二义性，噪声伪标签会造成“语义崩溃”，使模型难以收敛。CAESAR 只在训练阶段使用离线三维分割模型作为语义锚点，把语义能力蒸馏进配准器；推理时移除语义组件，因此零额外推理开销。

#### 方法动机分析
驱动力是让稀疏 LiDAR 配准在没有人工对应标注时仍能稳定学习。自蒸馏方法容易形成错误伪标签不断强化的循环；直接融合语义又要求在线运行分割模型，增加部署成本。作者观察到无监督配准特征本身含有可利用的语义结构，因此核心思路是用外部语义稳定并内化这一结构，而不是把语义网络放进推理链路。

#### 方法设计详解
输入两帧点云，主干提取几何特征并预测候选对应。DCRM 在 Top-K 候选中重匹配，以几何显著性和语义相似度的乘积评分：前者使用局部协方差特征值比率，后者使用加权 Jensen–Shannon 散度。SGLM 先用每个 mini-batch 的小型 SATR 适配器激活稳定特征，再由 RLS 根据空间相容性生成并选择变换假设。适配器联合优化探索正则、重排序和重建损失；SPD 要求学生从几何上下文重建语义嵌入，训练目标为原配准损失加 λ_SPD 蒸馏损失。训练采用逐步增大帧间隔、3–5 次 EM 式适配和学生 EMA。推理仅保留配准主干，约 0.16 s/pair。

#### 方法对比分析
与 INTEGER、EYOC 的自蒸馏不同，CAESAR 引入训练期外部语义锚点并用双线索重排候选，减少自指式伪标签腐蚀。与推理期语义融合相比，它将语义能力转化为几何上下文预测，适合无语义标注且强调实时性的 LiDAR 配准；代价是训练仍依赖预训练分割模型。

#### 实验分析（精简版）
在 64 线 KITTI 与 32 线 nuScenes 上，CAESAR 的 mRR 分别为 86.5% 和 79.5%，对应 INTEGER 的 84.0% 和 63.1%；KITTI 训练、nuScenes 测试时领先 11.5%。去掉 DCRM、SATR 或 SPD 会使 mRR 下降约 1.4–1.7%，训练时间较 INTEGER 减少 17.2%。即使分割教师 mIoU 只有 43.1%，方法仍达到 85.4% mRR。结果支持其抗稀疏和跨域能力，也说明各模块具有互补作用；但仍不能排除对分割教师质量的依赖，且实验没有覆盖更极端的传感器变化。

#### 实用指南
论文表示代码将开源，但当前未给出可核验仓库链接。复现需准备 KITTI/nuScenes、24 GB RTX 3090，并实现 progressive training、batch 级 SATR、RLS 假设选择、SPD 和 EMA；评估 RR、RRE、RTE、mRR。论文未完整给出学习率、batch size、epoch 和全部依赖，迁移到新传感器时应重新训练主干与适配器，并检查分割语义嵌入的域适配性，同时单独核验在线延迟和显存占用。

#### 总结
核心思想：语义训练锚定，配准推理零开销。
1. 几何主干产生候选对应。
2. DCRM 融合几何与语义重排。
3. SATR/RLS 挖掘稳定伪标签。
4. SPD 将语义能力蒸馏入学生。
5. 部署时移除语义分支输出变换。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15228v1)
- [arXiv](https://arxiv.org/abs/2609.15228v1)

---

<a id='2609.15162v1'></a>
## [LieSpline-DP: Lie-Group B-Spline Diffusion Policy for Smooth Robot Manipulation](https://arxiv.org/abs/2609.15162v1)

**Authors:** Erxuan Xie, Bang Liu, Pingyun Nie, Xingkai Liu, Zhuang Fu, Bo Zhang

**Published:** 2026-09-14

**Categories:** cs.RO

**Abstract:**

Diffusion Policy (DP) is a powerful Learning from Demonstration (LfD) method for robotic manipulation, yet it suffers from discontinuous and non-smooth trajectories. Spline-based action representations promote smooth motion within individual action chunks, but existing spline-based methods neither guarantee cross-chunk $C^2$ continuity nor account for the group structure of $\mathrm{SE}(3)$. We therefore propose LieSpline-DP, a Lie-group B-spline diffusion policy that generates end-effector trajectories directly on $\mathrm{SE}(3)$ and couples consecutive plans by sharing their boundary control poses, ensuring $C^2$ continuity throughout the entire planned trajectory. Across three real-robot tasks, LieSpline-DP produces lower trajectory jerk and higher task success rates than the DP baseline. The gains are particularly pronounced in real-world tasks involving liquids and flexible objects: in our real-robot experiments, LieSpline-DP achieved a 100% success rate on both pouring and bucket hooking, whereas the DP baseline achieved only 10% and 30%, respectively.

### 论文解读

#### 摘要翻译

扩散策略（DP）擅长从示范学习机器人操作，但预测的动作块切换不连续。现有样条表示只能保证块内平滑，既缺少跨块 C^2 连续性，也忽略 SE(3) 的群结构。LieSpline-DP 在 SE(3) 上直接生成末端轨迹，通过共享相邻计划边界控制位姿实现全程 C^2 连续，在倒水和挂钩提桶中分别达到 100% 成功率，而 DP 只有 10% 和 30%。

#### 方法动机分析

DP 循环预测离散 action chunk，异步执行时新旧块连接处可能出现速度、加速度突变，导致 jerk、跟踪误差，液体和绳索任务尤其敏感。欧氏插值也不符合旋转的 SE(3) 几何。论文假设把连续性写进动作表示，可减少对后处理的依赖；但紧凑样条可能限制高频精细修正。

#### 方法设计详解

输入是图像、机器人状态、当前末端位姿，输出是可采样执行的位姿曲线。示范先拟合为控制位姿 Q，轨迹写成 T_Q(s)=Q_0∏Exp(βΩ)，相邻点的 Ω 用 Log 表示，三次样条（p=3）天然提供 C^2 连续。扩散不直接在绝对位姿上运行，而把控制点变换到锚点 T_n^obs 的切空间 z=Log((T_n^obs)^−1Q)，去噪后再用 Exp 和群乘法恢复 SE(3)。Transformer 用全观测 cross-attention 融合视觉与状态，用块因果 self-attention 处理控制点；inpainting 固定上一计划末端 K=3 个点，只预测未来 F=8 个点。总控制点 H=11。训练损失为位姿扩散损失加夹爪损失 L_pose+λ_gL_grip；仿真 20 Hz、实机 15 Hz，并采用 S=2、E=4 的时间配置。

#### 方法对比分析

DP 输出离散路点且无显式跨块约束；BEAST、ABPolicy 虽用欧氏 B 样条，却主要依赖终点锚定或重拟合。本文的创新在于同时利用 SE(3) Log/Exp、共享前缀和样条解析性质，直接获得几何一致且严格 C^2 连续的轨迹，而不是另加滤波器。它适合对平滑性敏感的液体、柔性物体操作；追求尖锐快速修正的任务则可能更偏好离散表示。

#### 实验分析（精简版）

Robomimic 仿真中，相比 DP，平移 p95 jerk 降低 80.4%–80.7%，旋转 jerk 降低 85.8%–85.9%；Lift、Can 成功率约持平，但 Square、Tool Hang 略低，体现平滑与精细表达的权衡。UR5e 实机中，倒水成功率 100% 对 10%，挂钩提桶 100% 对 30%，方块堆叠 80% 对 55%。这些对比支持稳定性提升，但任务和骨干范围仍有限，不能外推到所有操作场景。

#### 实用指南

论文未明确给出可核验的代码仓库链接，开源状态未说明。复现需把离散示范拟合成 SE(3) 控制点，实现切空间扩散、固定前缀 inpainting 和样条采样，并同时测成功率与 jerk。迁移时要替换传感器/状态编码与末端运动学接口，重新拟合和训练；改变控制频率还需重调时间尺度，并验证碰撞、关节限位以及夹爪时序。学习率、batch size、epoch 与硬件资源未说明。

#### 总结

核心思想：李群样条让扩散动作连续。

1. 示范拟合为 SE(3) 控制点。
2. 在锚点切空间扩散去噪。
3. 固定旧计划前缀并预测未来点。
4. lift 成样条轨迹并采样执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15162v1)
- [arXiv](https://arxiv.org/abs/2609.15162v1)

---

<a id='2609.15213v1'></a>
## [X-WBC: A Cross-Embodiment Foundation Model for Humanoid Whole-Body Control](https://arxiv.org/abs/2609.15213v1)

**Authors:** Juntong Zhang, Chun Gu, Li Zhang

**Published:** 2026-09-14

**Categories:** cs.RO

**Abstract:**

Scaling humanoid whole-body control toward general-purpose deployment requires large human motion corpora and training experience shared across robot bodies. Existing methods usually train one policy per robot, leaving motion experience isolated across embodiments. We introduce X-WBC, a cross-embodiment foundation framework that separates relatively shared human motion semantics from embodiment-specific physical execution. Human-centered command tokens align full human motion, robot reference motion, and sparse VR observations. A causal Transformer learns reusable temporal structure from mixed multi-robot rollouts, while lightweight robot-specific modules map the shared representation to each robot's proprioception and action space. Across nine simulated embodiments, external motions, and four real robots, experiments show that joint training improves tracking, the aligned representation supports consistent control across command sources, and the learned policy remains competitive beyond the training corpus. These results support heterogeneous humanoids as joint data sources and establish cross-embodiment joint training as a practical route toward whole-body control foundation models.

### 论文解读
#### 摘要翻译
X-WBC 是面向人形机器人全身控制的跨具身基础模型。论文指出，传统 WBC 通常“一机一策”，不同机器人的训练经验难以共享；但动作意图具有跨具身共性，具体关节执行才受形态和动力学约束。作者联合训练多种人形机器人，并统一人类动作、机器人参考动作与稀疏 VR 指令。

#### 方法动机分析
核心矛盾是共享运动语义与机器人特定执行被混在同一策略中。X-WBC 假设同一动作在不同机器人、不同输入密度下应有相同潜在意图，从而共享语义主干，同时保留轻量的机器人适配模块。它主要面向人形机器人，新平台仍需适配训练。

#### 方法设计详解
输入包括三类指令：22 关节、264 维的人类动作；重定向为 14 个身体部位、168 维的机器人参考；仅含头部和四肢 5 个关键点、60 维的 VR 信号。人类与 VR 编码器共享，机器人参考编码器按机器人区分。状态含关节位置/速度、基座角速度、重力和上一动作，共 102 维。因果 Transformer 读取 32 帧历史，输出共享动作 token；机器人状态编码器和动作解码器再将其映射到各自关节空间。训练用 PPO，混合 9 种机器人，并加入不同指令 token 的成对 MSE 对齐损失。部署时以 50 Hz 推理，KV cache 只更新当前帧。

#### 方法对比分析
相比 H2O、OmniH2O 等单机器人 WBC，X-WBC 的差异在于跨具身联合训练、统一指令语义空间和机器人专属输入/输出适配。Transformer 利用时序上下文，共享骨干复用运动经验，适配器吸收关节数和动力学差异；因此更适合多个人形平台共享数据，但不能直接保证对四足或机械臂零样本迁移，也不能消除新平台的动力学标定成本。

#### 实验分析（精简版）
论文用约 200 小时 BONES-SEED 动捕数据训练，并在 100STYLE 的 100 种风格上评估。G1 成功率为 98.60%，高于单机器人训练的 97.69%；H2 为 93.22%，高于单机的 92.22%。未见风格的 VR 控制成功率达 91.13%，超过 TWIST 的 75.62%。跨具身 token 检索 Recall@1 为 61.3%–76.1%，远高于随机的 0.17%–1.04%；去掉对齐损失后 G1 降至 95.4%，显示该机制重要。验证仍主要覆盖仿真平地和有限接触。

#### 实用指南
复现环境为 Isaac Lab，训练约需 8 张 H100、2 天；需准备动捕数据、机器人模型和重定向流程，为每个平台定义状态与动作适配器，再进行多机器人 PPO。部署可使用 5 点 VR 输入和 KV cache。论文给出项目网站与演示，但未明确完整代码、权重及依赖的开源许可。迁移新平台时需替换重定向和适配模块并重训，复杂地形还需额外验证。

跨平台复现还应统一动作切片、奖励尺度与评估频率，确保成功率比较公平。

#### 总结
共享语义，适配具身

1. 多源动作编码为统一 token。
2. Transformer 融合 32 帧历史。
3. PPO 联合训练并用 MSE 对齐意图。
4. 专属解码器把意图落到各机器人关节。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15213v1)
- [arXiv](https://arxiv.org/abs/2609.15213v1)

---

<a id='2609.15921v1'></a>
## [Touch2Trace: Tactile-Driven Imitation Learning for Dexterous Cable Tracing](https://arxiv.org/abs/2609.15921v1)

**Authors:** Matteo Grimaldi, David Klee, Ziling Chen, Tong Jian, Wonju Lee, Wenjie Lu, Tao Yu, Saleh Nabi

**Published:** 2026-09-14

**Categories:** cs.RO

**Abstract:**

Dexterous manipulation of deformable objects demands continuous fingertip-level regulation of pressure, friction, and incipient slip. We study one of the most challenging cases: dexterous cable tracing, feeding a cable through the hand with repeated pinch-and-curl motions of the thumb and index finger. We introduce Touch2Trace, a tactile-driven imitation-learning system for this task, and provide, to our knowledge, the first systematic real-world characterization of how encoder pretraining, control rate, temporal context, and spatial resolution each shape policy performance. The winning learning recipe combines a tactile encoder pretrained for a custom 32 x 32 piezoresistive sensor (TacV5) via self-supervised learning with a lightweight transformer policy trained on teleoperated demonstrations via behavior cloning, deployed at 60 Hz on a Tesollo DG-5F hand. Tactile feedback without vision or explicit cable-state estimation significantly improves tracing performance versus a proprioception-only baseline: from 0.2 cm to 20.1 cm mean distance and 0% to 93% success rate, with zero-shot transfer to unseen cables and routing conditions. The results quantify the influence of key parameters in tactile-driven systems for reliable dexterous deformable object manipulation.

### 论文解读
#### 摘要翻译
本文提出 Touch2Trace，用触觉和本体感觉驱动灵巧手完成线缆跟踪，不依赖视觉或显式线缆状态估计。系统以 TacV5 高密度指尖触觉为核心，结合自监督触觉表征与时序模仿学习，并系统研究控制频率、时间上下文和空间分辨率对性能的影响。

#### 方法动机分析
线缆是变形体，压力、摩擦和滑动状态持续变化；仅凭关节角难以知道何时会滑脱，视觉也难稳定恢复接触状态。作者假设固定手部通过拇指和食指重复 pinch-and-curl，短时高分辨率压力序列便足以支持闭环调整。问题边界是固定工作空间内的跟踪：手臂不移动，且传感器主要测法向力。

#### 方法设计详解
策略训练时冻结触觉编码器，推理以60 Hz运行。
两指使用32×32 TacV5阵列（240 Hz采样，策略下采样至60 Hz）。每帧触觉图切为4×4 patch，送入8层ViT-MAE得到128维嵌入；编码器先在约200万帧模拟和真实触觉数据上以60–80%遮盖率自监督预训练，策略学习时冻结。策略接收最近15帧、共250 ms的两指触觉嵌入和8维关节位置，经三层因果Transformer建模接触动态，GMM动作头预测5个模式的联合分布，输出8维绝对关节位置。行为克隆最大化演示动作的条件似然；多峰分布可表示不同接触情形下的合理动作。作者以1D U-Net Diffusion Policy作对照。

#### 方法对比分析
创新在于把冻结触觉SSL、短时因果建模和GMM动作分布组合成适合小样本、高频闭环的方案，并将触觉的“配方”作为可验证因素。60 Hz保留快速反馈，15帧捕获滑动而不过度延迟，32×32保留接触细节。相较本体感觉策略，方法获得接触信息；相较扩散策略，GMM推理更适合实时控制。它更适合接触丰富、演示可重复且视觉受限的灵巧操作。

#### 实验分析（精简版）
作者仅用约10.1分钟、12段远程操作演示训练USB-0环形路径，并测试未见USB-1、USB-2、Ethernet及直线路径。仅本体感觉成功率为0%，跟踪距离低于0.2 cm；加入触觉后SR@10cm达到93%、SR@20cm为57%，平均跟踪距离20.1 cm。消融显示60 Hz、15帧历史和32×32输入优于较低频率、短/长上下文和低分辨率；扩散策略推理约32 ms，实时性不及TF-GMM。固定手部、仅法向力和未覆盖复杂缠绕是主要局限。

#### 实用指南
训练数据约为10.1分钟远程演示，推理控制频率为60 Hz。
复现需高密度指尖触觉、8自由度灵巧手、60 Hz控制、15帧历史、冻结ViT-MAE及5分量GMM。预训练约200万帧，遮盖率60–80%；位移评估使用独立机械编码器。论文提及robomimic等开源组件，但未明确提供本项目代码仓库；TacV5为内部硬件。迁移到其他手型需重新标定触觉布局、动作空间并收集演示；若要完成长距离或动态重新抓取，还需加入腕臂运动和重抓取策略。

#### 总结
核心思想：高密度触觉替代视觉跟踪线缆
1. ViT-MAE把压力图编码成触觉状态。
2. 15帧序列捕获接触与滑动动态。
3. 因果Transformer-GMM输出多峰关节动作。
4. 60 Hz闭环执行捏合卷动并推进线缆。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.15921v1)
- [arXiv](https://arxiv.org/abs/2609.15921v1)

---

