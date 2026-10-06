time: 20261006

# Arxiv Computer Vision Papers - 2026-10-06

## Executive Summary

## 执行摘要

本期 10 篇论文集中体现出两个方向：一是让机器人从固定示范走向可恢复、可扩展的闭环交互；二是把感知、记忆、路线和安全评估放进更接近真实部署的统一协议中。人形机器人方向覆盖 I-BFM 的物体接触行为基础模型、InterMimicGen 的自演化动作数据飞轮、Robotizing Human Videos 的接触保持视频迁移、AMF 的低延迟平滑控制，以及 TRABot 的语音—手势—面部协同。自动驾驶方向则关注长时域导航、4D 雷达规划价值和 SWIR 在困难环境中的互补性。

特别值得关注的是：I-BFM 将物体与接触动力学纳入共享行为表示，并在扰动和跌倒后保持恢复能力；InterMimicGen 通过“动作编辑—物理跟踪—成功筛选”持续扩大可执行示范覆盖；Odyssey 表明道路级路线遵循与及时换道准备不能由短时域开环指标替代；Radar2Plan 和 SWIR 分析都说明，传感器优势依赖规划架构及具体失效场景，而不是简单的模态排名。

正在形成的研究趋势包括：用物理验证或任务结果筛选生成数据，用显式状态绑定改善视觉语言模型的记忆与决策，用统一基准区分感知收益是否传递到规划和闭环控制，以及将安全 guards、拒绝行为和可审计记录纳入具身智能体评估。Inspect Robots 进一步说明，评估基础设施本身正在成为研究对象；MarvisNav 则展示了“把探索记忆投影到当前视野”这一轻量但有效的表示设计。

若时间有限，建议优先精读 I-BFM、InterMimicGen、Odyssey 和 MarvisNav：它们分别代表接触行为建模、自演化数据、长时域闭环驾驶评测和记忆表示四条互补路线。若关注真实机器人控制，再读 AMF 与 TRABot；若关注自动驾驶传感器和评测协议，则选择 Radar2Plan 与 SWIR 分析。

---

## Table of Contents

1. [Inspect Robots: Evaluating the Capabilities and Safety of Embodied AI](#2610.06306v1)
2. [I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning](#2610.06129v1)
3. [Odyssey: A Closed-Loop Benchmark for Long-Horizon Real-World Driving with Explicit Navigation Routes](#2610.06469v1)
4. [Radar2Plan: Benchmarking 4D Radar for End-to-End Open-Loop Ego-Trajectory Planning](#2610.06121v1)
5. [Analysis of SWIR Imaging Detection Performance Under Adverse Environmental Conditions for Autonomous Driving Systems](#2610.06596v1)
6. [Robotizing Human Videos with Physically Consistent Interactions](#2610.06137v1)
7. [MarvisNav: Making Memory Visible on Route Choices for Zero-Shot Object Navigation](#2610.06510v1)
8. [Talk, Render, Act: Integrating Social Gesture and Digital Face with Synchronized Speech for Conversational Humanoid Robot](#2610.06153v1)
9. [Adaptive Mean Flow for Responsive Closed-Loop Robot Control](#2610.06089v1)
10. [InterMimicGen: Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation](#2610.06850v1)

---

## Papers

<a id='2610.06306v1'></a>
## [Inspect Robots: Evaluating the Capabilities and Safety of Embodied AI](https://arxiv.org/abs/2610.06306v1)

**Authors:** Christopher Leet, Achu Menon, Sravanthi Machcha, Sabrina Zou, Aayushya Patel, Aditya Kumar Singh, Anish Kr Singh, Galaba Vamsi, Javin Ahuja, Sai Asish Yamani, Tushar Anand, Vedang Alle, Zihan Jack Zhang, Tzu Kit Chan, Jay Chooi

**Published:** 2026-10-05

**Categories:** cs.RO

**Abstract:**

General purpose language models are increasingly able to control robotic hardware. Understanding the capabilities and safety of these models when embodied is therefore increasingly important for understanding their societal impact and risks. To this end, we introduce Inspect Robots, a modular, open-source framework for developing and running evaluations of embodied agents. Inspect Robots pairs customizable, reusable abstractions for specifying physical evaluations and analyzing their results with infrastructure that automates evaluation setup, execution and termination. We demonstrate Inspect Robots by using it to evaluate the capabilities and safety of six policies based on frontier language models. Inspect Robots has seen significant early uptake, receiving nearly 100,000 downloads in the three months since its release.

### 论文解读

#### 摘要翻译
通用语言模型控制机器人硬件的能力日益增强。因此，理解这些模型具身后的能力与安全性，对认识其社会影响和风险愈发重要。为此，我们提出 Inspect Robots：一个用于开发和运行具身智能体评估的模块化开源框架。它将用于定义物理评估、分析结果的可定制、可复用抽象，与自动化评估设置、执行和终止的基础设施结合。我们通过评估六种基于前沿语言模型的策略展示该框架。发布后三个月内，Inspect Robots 下载量接近十万次，显示出显著的早期采用情况。

#### 方法动机分析
虚拟环境中的安全训练未必迁移到物理世界；新任务还要求快速搭建评估。已有工具分别覆盖基准、策略接入或自动执行，本文强调贯穿全流程的接口复用。核心假设是：可组合抽象能降低开发成本，同时容纳能力评分、拒绝行为与风险控制；它并不直接提升模型安全性。

#### 方法设计详解
输入是场景、指令、多个评分规则、时间预算和试验次数。多规则允许分别判断拒绝与减害行为。

策略定义为 π=(O,A,F)，其中 F:O×S→A⁺×S：观察与内部状态映射为动作序列及更新状态。推理时，直接预测动作的模型可直接接入；语言模型经适配器接收预处理观察、机器人操作说明、控制工具及终止工具，工具调用转为动作。

动作管理器在每个动作完成后决定继续序列，或用最新观察重新规划；随后 guards 对待执行动作修改或拒绝，必要时终止试验。机器人封装负责执行和复位。记录保存配置、观察、动作序列及终止原因；编辑器可写入标注，分析器只读，避免分析修改原始记录。本文无新训练损失。

#### 方法对比分析
AutoEval 重在自动执行、成功检测和重置，XPolicyLab 重在策略接口，RoboDojo 重在标准任务。本文贡献主要是工程抽象的系统组合：端到端接口减少重复实现，终止工具支持拒绝，多评分规则支持安全分析，可组合 guards 在研究风险与保护硬件间划界。

#### 实验分析（精简版）
能力评估用五个模型完成四项 RoboDojo 任务；安全评估用六个模型执行四项 RoboHarm 指令，每个模型每任务20次。

两项关键结果：Astra 比 GPT-5.6 Sol 的能力平均分高33个百分点；除玩偶伤害任务外，任何模型对其余危险任务的拒绝率均不超过5%。这提示能力提升不等于物理安全泛化，但不能证明框架改善了模型。无消融或框架间受控比较，任务少、单一机器人也限制外推。

#### 实用指南
论文称框架开源，提供 PyPI 统计链接，但未给仓库地址；模型权重和完整评估数据开放情况未说明。实验采用 I2RT YAM 双臂、视觉适配器及硬件限幅 guard，人工标注记录。预处理细节、提示词、采样参数和具体时间预算未说明。迁移需替换机器人观察/动作接口、复位逻辑、工具和 guards，并重写任务评分；是否重训取决于策略兼容性，论文未验证。

#### 总结
核心思想：以可复用接口贯通具身安全评估
1. 将场景、指令与多维评分封装为任务。
2. 通过策略适配器统一动作与拒绝接口。
3. 由动作管理器重规划，guards 约束执行。
4. 分离记录编辑与只读分析，复用评估流程。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06306v1)
- [arXiv](https://arxiv.org/abs/2610.06306v1)

---

<a id='2610.06129v1'></a>
## [I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning](https://arxiv.org/abs/2610.06129v1)

**Authors:** Ziqi Han, Yitang Li, Junhan Sun, Fanrong Dong, Yaojie Shen, Lei Ye, Zetong Jing, Yongqi Zhang, Yiming Zhang, Xue Wang, Hao Zhao

**Published:** 2026-10-05

**Categories:** cs.RO

**Abstract:**

Behavioral foundation models (BFMs) have recently shown that a single humanoid policy can support diverse whole-body control, but extending such generality to physical interaction remains challenging. We introduce I-BFM, to our knowledge the first BFM for humanoid-object interaction. Rather than relying on task-specific policies or reference tracking, I-BFM learns a shared representation of the coupled dynamics among the humanoid, objects, and their contacts using forward-backward representations and unsupervised reinforcement learning. Given a downstream task reward, the same policy can be directly conditioned on a latent command to execute closed-loop interaction without task-specific policy optimization. To improve interaction control over different time scales, we further train the policy with both short-horizon interaction targets and longer-horizon goal targets. A single I-BFM policy performs carrying, pushing, and kicking, while also supporting goal reaching, motion tracking, stylistic control, and long-horizon task chaining. More importantly, it remains effective after large deviations from nominal execution: on Carry, I-BFM achieves 94.3% nominal success and retains 89.3% success after robot falls, compared with 1.3% for a planning-based baseline. Real-world experiments on a Unitree G1 further demonstrate diverse loco-manipulation behaviors, rapid recovery from interaction failures and external disturbances, and task chaining without task-specific retraining.

### 论文解读

#### 摘要翻译
行为基础模型（BFM）已展示单一人形策略支持多样全身控制的能力，但推广到物理交互仍具挑战。我们提出 I-BFM，据我们所知，这是首个人形机器人—物体交互 BFM。它不依赖任务专用策略或参考跟踪，而以前向—后向（FB）表示和无监督强化学习，学习机器人、物体及接触耦合动力学的共享表示。给定下游奖励，同一策略直接接受潜在指令，实现闭环交互，无需任务专用策略优化。为改善不同时间尺度的控制，我们联合使用短期交互目标和长期目标训练。单一策略实现搬运、推动、踢动，并支持目标到达、动作跟踪、风格控制和长程任务串联。偏离正常执行后仍有效：搬运正常成功率为94.3%，机器人跌倒后为89.3%，规划基线仅1.3%。Unitree G1 实机实验进一步展示多样移动操作、交互失败和外部扰动后的快速恢复，以及无需任务专用重训的任务串联。

#### 方法动机分析
参考跟踪难以应对接触丢失，在线重规划存在延迟。核心假设是：将物体与接触纳入 FB 表示，按奖励检索行为，比追随固定轨迹更利于恢复。“零样本”指预训练后无需任务专用优化，不意味着无需交互数据。

#### 方法设计详解
输入为本体、物体位姿与速度、双手接触及相对位移；演员使用可部署观测历史，评论家可用特权状态。运动数据无物体时相关项置零，三类交互共享任务ID。

FB 以 \(F^\top B\) 近似未来状态访问量；联合条件判别器运动先验和辅助控制奖励训练，并非完全没有奖励信号。

新模块 LOGO 将后向特征投影到球面，分别取下一步和第八步目标，以球面对数映射表达方向与距离，替代易混淆动作意图的窗口平均。共享演员增加局部／目标残差损失，权重4:3、系数0.005；更新时冻结后向嵌入梯度。

推理以 \(z_r=P(\mathbb E[B(s)r(s)])\) 得到指令；根据距离、接触和进度切换阶段奖励，实现闭环控制。

#### 方法对比分析
相比身体中心 BFM，新增物体—接触建模；相比 OmniContact，不经规划轨迹调用跟踪器。FB 是既有组件，LOGO 是主要目标函数改造。任务串联仍需阶段逻辑，不等于自动规划任意任务。

#### 实验分析（精简版）
MuJoCo 中每项三组、每组100回合，最长60秒，成功距离阈值0.2米。关键证据：
- 搬运跌倒后成功率89.3%，OmniContact为1.3%，支持恢复优势。
- 移除LOGO后成功率由94.3%降至51.0%，但未拆分验证各组成项。

搬运误差表格为0.15米，正文为0.11米，存在冲突；部分基线初始化和终止规则不同。实机主要为定性证据，不能确认广泛物体泛化。

#### 实用指南
[项目页](https://iamhardworking.github.io/I-BFM/)已提供，代码、权重及数据开放状态论文未说明。实机依赖动捕，策略50Hz，测试箱体边长0.35米、重0.7千克。网络规模、数据量及训练算力未说明；复现须核对评估例外。迁移机器人需适配观测、动作和接触接口，并重新训练验证。

#### 总结
核心思想：用奖励寻址交互与恢复行为
1. 联合编码机器人、物体和接触的FB表示。
2. LOGO分离即时接触意图与远期目标。
3. 阶段奖励检索潜在指令，共享策略闭环执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06129v1)
- [arXiv](https://arxiv.org/abs/2610.06129v1)

---

<a id='2610.06469v1'></a>
## [Odyssey: A Closed-Loop Benchmark for Long-Horizon Real-World Driving with Explicit Navigation Routes](https://arxiv.org/abs/2610.06469v1)

**Authors:** Jungho Kim, Hongjae Shin, Seunghoon Yu, Heecheol Yoo, Myeongjun Kim, Jiyong Oh, Donghyuk Kwak, Seunghyeop Nam, Haesung Oh, Hyunju Kim, Hyungchan Cho, Jaehyun Park, Soo Won Seo, Jun Won Choi

**Published:** 2026-10-05

**Categories:** cs.RO, cs.AI

**Abstract:**

Closed-loop evaluation of end-to-end driving requires continuous rollouts that reveal how earlier decisions affect subsequent driving. However, existing benchmarks evaluate only short segments and fail to capture later consequences. Ambiguous directional commands also obscure the intended navigation objective. We introduce Odyssey, a closed-loop benchmark for long-horizon driving comprising 100 scenarios, each reconstructed from a 100-second nuPlan driving log to preserve the context of navigation maneuvers and traffic interactions. To provide a consistent navigation objective, Odyssey replaces directional commands with explicit standard-definition (SD) map routes that specify which roads to follow, while sensor-based planning determines local driving actions. Throughout these rollouts, diffusion-based refinement of 3DGS-rendered images reduces rendering artifacts along the ego trajectory. To assess how effectively planners follow these routes and prepare for upcoming maneuvers, we introduce SD Route Compliance and Pre-Lane Change Score. These assessments are complemented by RouteDS, which extends the Driving Score with penalties for SD-route deviations and failed lane preparation. We adapt state-of-the-art planners, including vision-language-action (VLA) models, and evaluate their navigation performance using these metrics. Odyssey highlights open questions in route representation and integration for E2E driving. Benchmark code and adapted baselines will be released publicly.

### 论文解读

#### 摘要翻译
Odyssey 是含100个场景的长时域闭环驾驶基准，每个场景由100秒 nuPlan 日志重建，保留导航动作和交通交互。它以显式标准精度地图（SD）路线替代含糊方向指令，局部动作仍由传感器规划；扩散细化用于减少自车轨迹上的3DGS图像伪影。基准提出 SD Route Compliance、Pre-Lane Change Score 和加入路线偏离及车道准备惩罚的 RouteDS，并适配包括VLA在内的规划器。

#### 方法动机分析
短片段无法揭示“提前换道—稍后转弯”的因果链，右转指令也不能区分连续出口；偏离日志视角还会损害渲染质量。方法假设长时域、明确道路目标和可靠视觉反馈能更真实地评估导航，但道路级路线不应替代车道级感知。其边界是路线遵循、提前准备与局部驾驶可能互相牵制。

#### 方法设计详解
日志轨迹经HMM匹配OpenStreetMap并人工检查生成路线折线；规划器每步接收前方120个、间隔1米的点，视觉基线每五点聚合token并用交叉注意力融合。OmniRe重建背景、车辆和行人，以覆盖感知采样提升低观测区域权重，并用地面高斯约束几何、时间条件外观表达信号灯变化；Fixer单步修复渲染且不改仿真状态，微调使用像素、LPIPS和Gram损失。轨迹经10 Hz LQR及自行车模型推进后重新渲染，交通按自车进度分段同步，从而保持连续交互而非只播放原日志。

#### 方法对比分析
相较NeuroNCAP、HUGSIM的短窗口，核心创新是长日志、SD路线、分段交通同步和导航指标的联合评估；3DGS、扩散与控制器属于既有组件。SD路线明确道路级目标，Pre-Lane Change Score 则补足“到达路线但准备太晚”的缺陷。它适合拥有多相机、轨迹、地图及动态目标记录的驾驶数据。

#### 实验分析（精简版）
这些结果同时暴露长期导航的评估边界。
100个场景评估五种规划器。非反应交通下，SD路线使DiffusionDrive的RouteDS由25.6升至35.9、SDC由59升至100，但四种规划器车道准备准确率下降，说明道路遵循不等于提前选道。ReCogDrive经GRPO后，NAVSIM PDMS由86.8升至90.4，闭环RouteDS却由37.7降至21.5，表明孤立轨迹收益不保证连续驾驶收益。

#### 实用指南
论文承诺发布代码和基线，但当前材料不能确认已开源。重建使用八相机、10 Hz和14万次优化；基线在NAVSIM训练并预测4秒轨迹。迁移需重新匹配路线、校准相机、重建场景并适配Fixer。微调使用全部100场景，不能作为跨场景泛化证据；具体RouteDS罚值和终止细则未说明，精确复现仍需补齐。

#### 总结
核心思想：明确路线检验长期导航
1. 将长日志匹配为道路级路线，保留局部决策空间。
2. 用覆盖感知重建和扩散细化提供视觉反馈。
3. 按自车进度同步交通，连续执行并重渲染。
4. 联合路线遵循、提前选道与累计违规评价导航。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06469v1)
- [arXiv](https://arxiv.org/abs/2610.06469v1)

---

<a id='2610.06121v1'></a>
## [Radar2Plan: Benchmarking 4D Radar for End-to-End Open-Loop Ego-Trajectory Planning](https://arxiv.org/abs/2610.06121v1)

**Authors:** Ling Yao, Yichun Xiao, Jin Jin, Yihan Zhang, Fangqiang Ding

**Published:** 2026-10-05

**Categories:** cs.RO

**Abstract:**

Adverse weather and poor illumination remain major challenges for robust ego-trajectory planning in mobile autonomy. 4D radar offers reliable sensing under adverse conditions and direct radial-velocity measurements. However, sensing robustness does not necessarily translate into robust downstream planning, while existing 4D radar benchmarks focus primarily on perception rather than trajectory planning. We present Radar2Plan, a modular benchmark for open-loop ego-trajectory planning using real-world 4D radar data. Radar2Plan connects sensor encoders, scene representations, and planning heads through common interfaces, enabling controlled comparisons between different sensor and planner configurations. Using DSERT-RoLL and MAN TruckScenes, we evaluated seven combinations of camera, 4D radar, and LiDAR in four representative planning baselines and 12 weather and illumination conditions under a unified protocol. To our knowledge, Radar2Plan is the first benchmark dedicated to evaluating real-world 4D radars for autonomous driving planning. Experiments show that 4D radar alone supports competitive ego-trajectory planning and robust performance under adverse conditions. Sensor-configuration comparisons further demonstrate its complementary value to other modalities, while revealing dependencies on sensor combination and planning architecture. The modular design also supports additional datasets, sensing modalities, and planners, providing a flexible foundation for future research on radar-based autonomous driving planning.

### 论文解读

#### 摘要翻译
恶劣天气和低照度下，自主系统难以稳健规划自车轨迹。4D雷达能可靠感知并直接测量径向速度，但感知鲁棒性未必转化为规划鲁棒性，现有基准也主要评估感知。Radar2Plan是首个面向真实4D雷达规划的模块化开环基准，统一连接传感器编码器、场景表示和规划头，在DSERT-RoLL与MAN TruckScenes上比较相机、4D雷达、LiDAR的七种组合、四类规划基线及十二种天气和光照条件。结果显示雷达单独即可获得有竞争力且适应恶劣条件的规划性能，但融合收益依赖传感器组合和规划架构。

#### 方法动机分析
论文要回答“感知更可靠是否使规划更可靠”。既有雷达工作多停留在检测，单一架构又无法区分模态收益与模型偏好。核心假设是：固定数据接口、编码器和评估协议，才能公平比较模态对规划的贡献。问题边界是开环拟合记录轨迹，不能直接证明闭环安全。

#### 方法设计详解
输入→表示→输出：数据适配器对齐传感器与位姿，把未来轨迹变换到自车坐标，形成6秒、5Hz的30个二维点。相机使用冻结DINOv3；雷达和LiDAR使用独立PTv3，并融合当前帧、运动补偿前帧及自车运动历史。场景适配器输出BEV或查询令牌：缺失BEV置零，查询式融合仅拼接有效令牌，从而让同一规划头接收不同模态。四类规划头分别回归航点、预测并评分多候选、从固定词表分类或用扩散细化，训练目标对应L1、回归加模式分类、交叉熵及分类加L1；推理选最高分轨迹，以Top1ADE衡量平均欧氏误差。统一训练80轮，AdamW学习率和权重衰减均为10⁻⁴，批量16、随机种子42。

#### 方法对比分析
相比只在单架构使用雷达的SpaRC-AD，以及不把雷达输入规划器的TruckDrive，Radar2Plan的新意是跨模态、跨架构的规划评估协议和模块解耦，而非新雷达编码器。四个规划器是适配版本，适合研究模态互补性；但组合效果受规划头和数据条件限制，不能把工程统一接口视为算法创新本身。

#### 实验分析（精简版）
实验按序列或场景隔离训练测试。两数据集、四种架构中雷达均是最佳单模态：MAN上稀疏查询模型ADE为1.96米，相机为5.78米、LiDAR为6.11米；加入雷达的24组比较有22组改善。并非总能增益：DSERT词表模型从相机加LiDAR的3.06米退化到三模态的3.37米。证据支持雷达的单模态优势和条件互补性，但仅单随机种子、无闭环安全指标，也缺少去除自车状态的消融。

#### 实用指南
复现需保持数据划分、位姿匹配、时间对齐、点数上限（雷达2048、LiDAR4096）及各规划头的检查点选择。论文未说明代码、权重和算力是否开源。迁移到其他机器人或数据集时，应替换标定与适配器、重建轨迹词表并重训规划模块，同时重新核查天气、光照和传感器覆盖差异。

#### 总结
核心思想：统一协议检验雷达规划价值
1. 将异构日志对齐到同一坐标和时间域，生成自车轨迹目标。
2. 用场景适配器连接雷达、相机、LiDAR与不同规划范式。
3. 以统一最高分轨迹评估，并按环境条件配对测量雷达收益。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06121v1)
- [arXiv](https://arxiv.org/abs/2610.06121v1)

---

<a id='2610.06596v1'></a>
## [Analysis of SWIR Imaging Detection Performance Under Adverse Environmental Conditions for Autonomous Driving Systems](https://arxiv.org/abs/2610.06596v1)

**Authors:** Rohan Mehra, Alexandre Riffard, Yannis Loumouamou, Mathieu Labussière

**Published:** 2026-10-05

**Categories:** cs.CV, eess.IV

**Abstract:**

Short-wave infrared (SWIR) imaging has emerged as a promising modality for autonomous driving, yet its practical benefits over RGB remain poorly characterized across diverse conditions. This paper presents a systematic comparative study of paired RGB and SWIR object detection on the RASMD dataset, covering four weather conditions and two real-time detection architectures, with various fine-tunings evaluated against a unified ground truth. Overall, RGB demonstrates comparable or superior performance in most scenarios, while RF-DETR exhibits greater robustness across varying conditions. Beyond aggregate metrics, we propose a sensor-dominance mining framework that combines multi-model agreement with targeted manual inspection to identify scenarios where one sensing modality provides more reliable detections using largely unannotated paired data. This analysis reveals that SWIR offers clear advantages in four safety-critical situations, including windshield glare, water droplets on the windshield, low-contrast object visibility, and long-range vehicle detection. The findings suggest that SWIR should be viewed as a complementary modality that enhances perception in rare but challenging conditions. The datasets will be available upon request, and all code and trained model weights are publicly released at https://github.com/comsee-research/swir-adverse-env-analysis.

### 论文解读

#### 摘要翻译
短波红外（SWIR）是自动驾驶的潜力模态，但相对RGB的实际收益尚不清楚。本文基于RASMD，系统比较四种天气、两种实时检测架构和多种微调策略，并用统一真值评估。RGB在多数场景表现相当或更好，RF-DETR跨条件鲁棒性更强。作者进一步结合多模型一致性与定向人工检查，从未标注配对数据中找出模态优势场景：SWIR在挡风玻璃眩光、水滴、低对比度目标和远距离车辆中有明确优势。因此SWIR更适合作为罕见困难条件下的互补模态。

#### 方法动机分析
天气平均指标会掩盖局部、短暂却危险的RGB失效，也难以解释传感器何时真正创造安全收益。核心假设是：跨模型一致的独有检测可筛出模态优势候选，再由人工排除幻觉。研究目标是发现检测互补性，而非证明SWIR全面替代RGB或实现自适应融合。

#### 方法设计详解
输入为约10万组对齐RGB–SWIR图像；SWIR按99%置信区间裁剪并拉伸直方图，目标只要在任一模态可见便统一标注。RASMD-IP含1713对训练/验证图像和800对测试图像，每种天气200对。COCO预训练YOLOv8x、RF-DETR分别进行单模态、等比例混合和天气专用微调；还用20% Pix2PixHD生成SWIR、80%真实SWIR。混合训练仍逐图推理，不是双流融合。优势挖掘用三个投票组；同类框以IoU≥0.3匹配，仅一侧有未匹配框即投该模态票，冲突候选丢弃，再以至少1/2/3票分级判定。人工核验SWIR候选并保持至少10帧间隔，形成532对RASMD-HiLS。

#### 方法对比分析
相较单一天气或整体评估，本文的本质新增是“共识筛选—人工确认—场景验证”，并统一跨模态真值。检测器和图像翻译是标准组件，创新主要在分析协议与数据构建；它适合对齐良好、标注稀缺的成对传感器数据。

#### 实验分析（精简版）
测试按类别稀有度采样，并报告加权指标与1000次bootstrap AP。单模态微调RF-DETR的RGB/SWIR总体mAP为0.4374/0.3651，说明RGB总体更强；但未参与筛选的验证中，眩光集SWIR/RGB召回率为0.7204/0.3318，显示局部互补。局限是优势集主动筛选、不能估计自然发生率，且61.5%候选被丢弃，类别失衡与模型偏差仍影响结论。

#### 实用指南
代码与权重已公开（GitHub：comsee-research/swir-adverse-env-analysis），数据按需提供。复现需保持配对对齐、统一标注和时间去重；论文未说明学习率、轮数、硬件及完整依赖。迁移时需重新标定和微调检测器，并重做人工核验，不能直接沿用优势类别。

#### 总结
核心思想：用共识挖掘模态互补场景
1. 匹配跨模态检测框，提取独有目标。
2. 多组投票排除冲突，分级确定优势候选。
3. 人工核验并去重，归纳传感器失效场景。
4. 用未参与筛选的模型验证场景收益。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06596v1)
- [arXiv](https://arxiv.org/abs/2610.06596v1)

---

<a id='2610.06137v1'></a>
## [Robotizing Human Videos with Physically Consistent Interactions](https://arxiv.org/abs/2610.06137v1)

**Authors:** Ching-Lam Cheng, Shengfeng He, Bin Zhu

**Published:** 2026-10-05

**Categories:** cs.RO

**Abstract:**

Human videos offer scalable manipulation data, but the embodiment gap between human hands and robot manipulators limits their direct use. Existing video-editing methods replace hands with rendered robots, yet inaccurate interaction reconstruction and compositing can produce inconsistent grasps and implausible robot-object occlusions. We address these failures from two complementary physical aspects: interaction geometry and scene visibility. First, an interaction-aware contact reconstruction module combines hand-object segmentation with mesh-level contact prediction to recover dense 3D contacts, then converts them into temporally stabilized grasps for parallel-jaw grippers. Second, a depth-aware compositing module uses scene and robot depth to enforce physically consistent robot-object occlusions. The resulting videos preserve the interaction structure of human demonstrations in a robot-compatible form and are co-trained with robot demonstrations. Using identical human videos and robot data, we compare against robot-only training and the original Masquerade pipeline. Across four RoboTwin tasks and two Diffusion Policy visual encoders, our method achieves the highest average success rates, with especially strong gains under out-of-distribution scene variation. Real-world deployment further shows that the proposed co-training approach improves robustness to visual distractors when the task geometry is observable, while performance on depth-sensitive grasps remains limited by the single-camera setup.

### 论文解读

#### 摘要翻译
人类视频虽能提供大量操作数据，但人手与机器人操纵器的具身差异使其难以直接使用。本文从交互几何与场景可见性两方面改进将人手替换为机器人的视频编辑：交互感知接触重建结合手—物体分割和网格级接触预测，恢复稠密三维接触并转换为时间稳定的平行夹爪抓取；深度感知合成利用场景和机器人深度保证物理一致的遮挡。生成视频保留人类示范的交互结构，并与机器人示范联合训练。

#### 方法动机分析
外观像机器人并不代表交互正确：仅依手姿态重定向可能错失真实接触，直接覆盖渲染又会产生不合理遮挡。核心假设是保持接触关系与可见性，能让人类视频提供更可靠的机器人策略监督；但它无法补足单相机不可观测的几何。

#### 方法设计详解
训练阶段将编辑视频的路点监督与机器人示范动作损失联合；推理阶段沿视觉路点执行。
输入是RGB视频与重建的MANO手网格。VISOR稀疏标注经CaRe-Ego扩展为逐帧分割；物体掩码质心、手几何及投影送入LatentAct，输出778个手顶点的接触概率并阈值化。接触点分成拇指和非拇指两组，以质心代表夹爪两端；非拇指点还须满足法向与指向拇指方向的点积大于0.3，无合格点时回退原集合，无接触时使用指尖。以前一有效点权重0.3进行时间平滑，再以最近顶点确定对向手指。去手后由E2FGVI修复并估计场景深度，仅保留不被更近场景遮挡的夹爪像素，机器人主体不裁剪。编辑视频由预训练视觉编码器预测未来二维路点，与机器人示范的动作损失联合训练。

#### 方法对比分析
训练阶段将编辑视频的路点监督与机器人示范动作损失联合；推理阶段沿视觉路点执行。
相较原始Masquerade，新增接触约束重定向和深度遮挡裁剪，而非新策略网络。分割、接触预测、修复等多为已有组件，贡献在于把它们组合为交互保持流程。方法适合平行夹爪，但不等于保证动力学或力闭合。

#### 实验分析（精简版）
每项任务使用50条机器人示范，仿真ID训练并在含干扰物的OOD场景各评估100次。DP-ViT的平均OOD成功率由Masquerade的23.50%升至45.75%；去掉接触重建和深度遮挡后分别降至30.25%和35.75%，支持两模块的作用。但DP-Swin摇瓶OOD从36%降至21%，收益并不稳定。实机每设置仅10次，海绵OOD为8/10，对照7/10；箱子为2/10，且未报告多种子不确定性。

#### 实用指南
论文提供项目链接，但未明确代码与权重的开放状态。复现需对齐分割、手网格、相机投影和渲染坐标，并实现接触阈值、时间平滑及深度尺度对齐；具体深度模型配置、训练预算和人类视频数量未说明。迁移到其他机器人需替换抓取重定向与渲染，并用目标机器人的动作数据训练策略；换数据集则需补充分割支持。

#### 总结
核心思想：以接触和遮挡保持交互
1. 用物体质心引导手网格接触预测。
2. 经法向筛选与时间平滑提取对向抓取。
3. 比较去手场景与机器人深度，裁剪夹爪遮挡。
4. 以编辑视频路点监督辅助机器人策略学习。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06137v1)
- [arXiv](https://arxiv.org/abs/2610.06137v1)

---

<a id='2610.06510v1'></a>
## [MarvisNav: Making Memory Visible on Route Choices for Zero-Shot Object Navigation](https://arxiv.org/abs/2610.06510v1)

**Authors:** Jincheng Wang, Chi Pui Chan, Wei Zeng, Shuyang Zhang, Jianhao Jiao, Dimitrios Kanoulas

**Published:** 2026-10-05

**Categories:** cs.RO

**Abstract:**

When searching for an object, people choose their next move by considering both likely target locations and places already explored. The current view can cue place-associated memories, bringing target relevance and prior exploration into the same spatial context. In many zero-shot object navigation (ZSON) methods, however, vision-language models (VLMs) infer promising search areas from egocentric images, while exploration history is represented separately, e.g., as text or maps. This separation either requires an additional fusion step or leaves the correspondence between memory and route choices implicit for the VLM to recover. We instead make exploration memory directly visible on visual route choices. We propose MarvisNav, a ZSON framework that maintains a topological graph and projects candidate nodes together with their exploration states onto egocentric views as memory-bearing visual route choices. These states capture local exploration progress beyond binary visitation. By binding exploration state directly to each visual candidate, MarvisNav enables the VLM to jointly evaluate target relevance and exploration state without a separate post-hoc fusion or reranking stage. Without policy training, MarvisNav achieves state-of-the-art performance on HM3D (81.2% SR and 42.5% SPL), while remaining competitive on MP3D. It also outperforms representative VLM-based methods with far fewer VLM calls (e.g., 7.5% of WMNav). Real-robot experiments across diverse scenes further validate its practical deployability. Beyond MarvisNav, our study shows that memory representation shapes VLM decisions and ZSON performance, highlighting that effective memory use depends not only on its availability, but also on how it is represented. Code and project page will be available at \url{https://wangjincheng1998.github.io/MarvisNav/}.

### 论文解读

#### 摘要翻译
寻找物体时，人们同时考虑目标可能的位置和已探索区域。当前视野可唤起地点相关记忆，将目标相关性与探索历史置于同一空间语境。但许多零样本物体导航（ZSON）方法让视觉语言模型（VLM）从第一视角图像推断搜索区域，却以文本或地图单独提供历史，因此需要额外融合，或让模型隐式恢复记忆与路线的对应。我们提出 MarvisNav：维护拓扑图，将候选节点及其探索状态投影到第一视角，使路线选项直接承载记忆。这些状态刻画超越二元访问记录的局部探索进度，让 VLM 联合评估目标相关性与探索状态，直接完成高层选择，无需事后融合或重排序。不训练策略，MarvisNav 在 HM3D 达到最先进的81.2% SR、42.5% SPL，在 MP3D 上具有竞争力；其性能超过代表性 VLM 方法，调用量仅为 WMNav 的7.5%。多场景实机实验验证了部署能力。研究表明，记忆表示影响 VLM 决策和导航表现，有效利用记忆不仅取决于是否提供，还取决于如何表示。代码将于项目网站发布。

#### 方法动机分析
独立历史要求模型跨语言、地图与当前视野建立对应；外部过滤又可能覆盖语义判断。核心假设是：把“还有多少探索价值”直接绑定路线，比仅标记“来过”更有效。研究针对部分可观测的长期搜索，不解决全部感知误差。

#### 方法设计详解
RGB-D与位姿更新自由／占据／未知栅格，保留可达自由空间，骨架化生成约简 Voronoi 图（RVG）。端点按是否关联活跃前沿分为未探索、已探索；路口按是否存在直接相邻的前沿关联节点分为开放、已探索。

局部拓扑决定取景方向；相机投影与深度一致性检验剔除遮挡，将路径、候选字母及状态颜色叠加到图像。VLM 输出路线字母和目标可见标志；字母直接映射导航坐标，可见标志可降低当前视图的检测阈值。低层执行后更新地图。无新增训练目标；新机制是状态与选项绑定，而非建图或控制器。

#### 方法对比分析
相比 VoroNav，增加局部探索状态及第一视角绑定；相比 Mem2Ego 的独立访问标记，无需通过图像距离猜测对应；相比外部过滤，保留 VLM 最终选择权。适合具有可靠深度、位姿的物体搜索，多楼层扩展尚待研究。

#### 实验分析（精简版）
三套基准共5195回合。固定流程的 HM3D v0.2 消融中，绑定状态较访问标记将 SR/SPL 从76.5/40.6提升至81.2/42.5。共享 Gemini-2.5-Flash 时，仅需4115次调用，WMNav 为54515次，且性能更高。

但消融同时改变状态信息和绑定方式，不能独立归因；MP3D 两项指标均低于 RememNav，误检仍是主要失败来源。

#### 实用指南
[项目网站](https://wangjincheng1998.github.io/MarvisNav/)仅承诺发布代码，未确认已开源。仿真采用640×480 RGB-D、79°视场、500步预算和单张 RTX 4090。图剪枝、投影容差等阈值论文未说明。实机使用 Go2、Jetson Orin及远程 VLM；迁移需适配传感器、检测器和控制器，所用预训练低层策略未微调。

#### 总结
核心思想：将探索记忆绑定视觉路线
1. 从在线RVG及前沿生成局部探索状态。
2. 按拓扑取景，将状态绑定候选字母。
3. VLM联合判断语义价值与剩余探索价值。
4. 字母转为导航目标，执行后更新状态。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06510v1)
- [arXiv](https://arxiv.org/abs/2610.06510v1)

---

<a id='2610.06153v1'></a>
## [Talk, Render, Act: Integrating Social Gesture and Digital Face with Synchronized Speech for Conversational Humanoid Robot](https://arxiv.org/abs/2610.06153v1)

**Authors:** Jin Jiang, Kun Li, Jiancong Ma, Shengcai Liao

**Published:** 2026-10-05

**Categories:** cs.RO

**Abstract:**

Expressive humanoid interaction requires speech, facial animation, and body gestures to form a coherent response. However, many full-body humanoid robots produce speech and gestures without a visually expressive face, while talking-face animation and robot gesture generation are typically developed separately. We present Talk, Render, Act (TRABot), an agent-based framework comprising specialized agents for motion-atom construction, dialogue generation, motion planning, and facial animation. First, to produce natural and semantically meaningful gestures, we construct Robot-Ready Semantic Motion Atoms by segmenting long-form, G1-retargeted BEAT2 motion into units with natural gesture boundaries, human-verified communicative functions, and feasible trajectories. Second, to preserve semantic order and coordinate body motion with the spoken response, we introduce a Semantic-Conditioned Compositional Planner. Given an ordered semantic function sequence and an estimated response duration, the planner selects approved atoms to realize the longest feasible action sequence while accounting for transitions and neutral recovery. Finally, we deploy a Streaming Face-Speech-Body Integration system on a physical G1 humanoid, combining streaming dialogue audio, audio-driven facial animation, and semantically planned body motion in a unified real-time interaction loop. Quantitative and qualitative experiments demonstrate that TRAbot achieves the best overall performance among all compared conditions in terms of naturalness, expressiveness, and multimodal coherence.

### 论文解读

#### 摘要翻译
富有表现力的人形机器人交互需要语音、面部动画和身体手势形成连贯回应。然而，许多全身人形机器人能说话、做手势，却缺少视觉表现丰富的脸部；说话人脸动画与机器人手势生成通常也独立开发。本文提出 Talk, Render, Act（TRABot），一个由动作原子构建、对话生成、动作规划及面部动画专用智能体组成的框架。首先，将重定向至 G1 的 BEAT2 长动作分割为具有自然手势边界、人工核验交际功能和可行轨迹的机器人就绪语义动作原子，以产生自然且有意义的手势。其次，提出语义条件组合规划器，根据有序语义功能序列和预计回复时长，在考虑过渡及中立姿态恢复的情况下，选择获批原子组成最长可行动作序列，保持语义顺序并协调语音与身体运动。最后，在实体 G1 上部署流式脸部—语音—身体集成系统，将流式对话音频、音频驱动面部动画及语义规划动作纳入统一实时交互循环。定量与定性实验显示，TRABot 在自然度、表现力和多模态连贯性方面取得所有比较条件中的最佳整体表现。

#### 方法动机分析
面部可逐帧跟随音频，身体却须满足轨迹可行性、状态衔接与时长约束。核心假设是：人工核验的离散交际功能能连接语言意图和可执行动作。目标是回复级协调，而非逐词精确同步。

#### 方法设计详解
BEAT2 重定向后控制14个臂部关节；25 Hz 平滑副本用于检测，30 Hz 原轨迹用于执行。按关节活动范围归一化速度，以录制级自适应双阈值提取活动段，再寻找低速度、低加速度、稳定姿态边界。候选经关节、导数和接触检查；失败者缩幅、平滑、延时修复，随后人工审批，Qwen3-VL 提议标签并由人工复核。
在线输入为功能序列、预计时长及实测姿态。总成本包括原子、过渡、回中立和时间余量；动态规划求最长可行功能前缀，宽度32的束搜索选择组合。每次只执行首个原子，以最小加加速度轨迹衔接，再测状态重规划。共享音频驱动语音与 SoulX-FlashHead；没有提出新训练损失。

#### 方法对比分析
相较 RoboGesture 的语音—身体路线，本文增加持续动画脸部；相较 Haru 的多模态表达，重点是人形机器人可行轨迹和状态相关组合。创新主要在动作目录及约束规划，脸部与对话模型属于现成组件集成，并非联合生成模型。

#### 实验分析（精简版）
20人对12轮、五种条件排序：完整系统获203/240次偏好（84.58%），手势—语音排名从同目录随机动作的3.8625升至4.7708。200条固定回复中，相比功能匹配随机规划，每条原子数从1.451升至1.705，剩余预算从2.376秒降至0.993秒。证据支持语义组合价值，但端到端样本小、未报告显著性；动作多样性受目录限制。

#### 实用指南
代码、权重发布及硬件资源论文未说明。复现需保留双采样率、人工审查及 MuJoCo 验证；在线预算为预计语音时长的3倍，余量0.35秒，过渡限速1 rad/s，因此预算可行不等于动作在语音结束前完成。迁移机器人须重定向、替换限位和接触模型并重审目录；论文未说明需重训基础模型。

#### 总结
核心思想：以语义动作原子连接对话与执行
1. 自适应分段并修复轨迹，人工核验动作及交际功能。
2. 将回复映射为有序功能与时长预算，搜索最长可行前缀。
3. 实测状态衔接、逐原子重规划，与共享音频驱动的人脸并行执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06153v1)
- [arXiv](https://arxiv.org/abs/2610.06153v1)

---

<a id='2610.06089v1'></a>
## [Adaptive Mean Flow for Responsive Closed-Loop Robot Control](https://arxiv.org/abs/2610.06089v1)

**Authors:** Aksel Vaaler, Marco Job, Christian Holden, Olav Egeland

**Published:** 2026-10-05

**Categories:** cs.RO, cs.AI

**Abstract:**

Diffusion- and flow-based robot policies have recently become widespread in robotic Imitation Learning (IL) due to their high performance and ability to model continuous and multimodal distributions. However, the iterative denoising procedure used by these models introduces significant prediction latency, hindering high-frequency closed-loop robot control and leading to jittery, unstable motion when frequent updates to the robot's action predictions are used. Therefore, it is common practice to train models to predict chunks of actions that can be executed sequentially without feedback, even when this reduces responsiveness and may mean the most recent state information is not used. In this article, we present Adaptive Mean Flow (AMF), a flow-based IL method that enables smooth and responsive, fully closed-loop robot control. AMF uses Mean Flow, which is an accelerated form of Flow Matching (FM), to minimize prediction latency. To ensure smoothness and consistency across predictions, AMF uses a corrupted version of the trajectory from the previous step when predicting new robot actions, with the signal-to-noise ratio increasing over the time parameter of the trajectory. This discourages large changes in the prediction from one step to the next, while allowing freedom to adapt the predictions for future steps. We evaluate AMF across a wide range of simulated and real robot tasks and demonstrate significantly improved performance compared with baselines. Code: https://github.com/akselva/Adaptive-mean-flow-RoboticIL.

### 论文解读

#### 摘要翻译

扩散与流模型机器人策略近年来在机器人模仿学习（IL）中得到广泛应用，原因在于其高性能，以及对连续、多模态分布的建模能力。然而，这些模型采用的迭代去噪过程会带来显著的预测延迟，阻碍高频闭环机器人控制；当机器人频繁更新动作预测时，还会产生抖动和不稳定运动。因此，常见做法是训练模型预测动作块，使机器人能够在没有反馈的情况下依次执行这些动作，即使这会降低响应性，并可能导致最新状态信息未被使用。

本文提出 Adaptive Mean Flow（AMF），一种基于流的模仿学习方法，可实现平滑、响应迅速且完全闭环的机器人控制。AMF 使用 Mean Flow——一种加速的 Flow Matching（FM）方法——来降低预测延迟。为确保不同预测之间的平滑性与一致性，AMF 在预测新机器人动作时使用上一步轨迹的受扰版本，其信噪比随轨迹时间参数增大而增加。这抑制了相邻步骤之间预测的大幅变化，同时允许模型自由调整未来步骤的预测。作者在广泛的仿真及真实机器人任务上评估 AMF，展示了相较基线显著改善的性能。

代码链接：[github.com/akselva/Adaptive-mean-flow-RoboticIL](https://github.com/akselva/Adaptive-mean-flow-RoboticIL)。

#### 方法动机分析

**核心矛盾是响应性与跨预测一致性，而不只是推理速度。**

- **动作分块执行**：同一次预测内部通常平滑，但执行期间不能及时利用新观测。移动目标和外部扰动容易使原有轨迹失效。
- **逐步重新预测**：反馈及时，但从独立噪声重新采样可能切换行为模式，产生抖动；单纯将 FM 换成一步 Mean Flow 并不能解决这一问题。
- **统一强度的轨迹复用**：能够稳定运动，却可能过度锁定未来计划，妨碍绕行、换向等必要的策略切换。

AMF 的核心假设是：**近期动作需要继承，远期动作需要可修改性。** 因此，它结合一步生成与沿预测时域变化的轨迹扰动，让近期预测更接近旧计划、远期预测保留更大调整空间。

问题边界主要是连续动作控制下的延迟、平滑性和动态鲁棒性。论文没有证明安全稳定性，也没有验证跨机器人零样本泛化。另需注意：摘要关于信噪比变化方向的表述与正文机制不一致；正文明确采用“越远期噪声越大”的设计。

#### 方法设计详解

##### 1. 输入与序列表示

输入为近期观测序列 \(O_t\)，包含机器人状态，真实机器人实验还包含两个相机的 RGB 图像。模型预测：

\[
A_t=a_{t-T_o:t+T_p-1},\qquad T=T_o+T_p.
\]

序列不仅包含未来动作，还包含过去 \(T_o\) 步动作。过去动作虽然不再执行，但参与轨迹复用，作为历史一致性的隐式约束。

论文主要设置为 \(T_o=9\)、\(T_p=10\)，共 19 个动作 token。动作归一化、图像裁剪和数据增强细节未说明。

##### 2. 旧轨迹对齐，再施加分时域扰动

将上一步预测左移一位，末尾复制最后一个动作：

\[
A_{t^-}=\operatorname{Concat}
\left([A_{t-1}]_{1:T-1},[A_{t-1}]_{T-1}\right).
\]

这样，旧预测与当前预测对应相同的实际时间步。随后构造输入：

\[
A_t^S=S\odot A_{t^-}+(1-S)\odot e,
\qquad e\sim\mathcal N(0,I).
\]

其中 \(S=(s_0,\ldots,s_{T-1})\) 是逐动作流索引：

- \(s_i\) 越大，保留的旧轨迹越多；
- \(s_i\) 越小，重新采样的自由度越大；
- 历史部分保持固定索引，未来部分设计为逐渐降低索引。

**“自适应”不是学习一个在线调噪控制器**：推理时使用固定参数的时域调度，轨迹则随着最新观测不断更新。

论文给出的调度为：

\[
s_i=
\begin{cases}
s_a,&i<T_o,\\
s_a(s_b/s_a)^{\lambda_\gamma(i)},&i\ge T_o,
\end{cases}
\qquad
\lambda_\gamma(i)=
\left(\frac{i-(T_o-1)}{T-T_o}\right)^\gamma.
\]

作者报告使用 \(s_a=0.5,s_b=0,\gamma=1.5\)。

**关键复现疑点**：按上述公式直接代入 \(s_b=0\)，所有未来位置的指数均大于零，因此未来部分全部变成 \(s_i=0\)，而不是渐变调度。另一个噪声敏感性实验的文字说明又使用 \(s_b=1.0\)。这些设定存在不一致，实际实现需要核对代码，不能直接把印刷公式视为无歧义实现。

##### 3. Transformer 与逐 token 时间条件

策略网络采用 Transformer：

- 8 层、4 个注意力头、嵌入维度 256；
- 扰动后的动作作为输入 token；
- 观测经投影后通过交叉注意力提供条件；
- 每个动作对应的起止流索引 \((s_i,r_i)\)，通过 **feature-wise Adaptive LayerNorm-Zero** 注入各层。

与仅在输入端拼接或相加相比，逐层条件化允许网络反复根据每个动作的扰动程度调整计算。

真实机器人实验中，两路图像分别经 ResNet-18 编码器处理，并与策略端到端训练。Transformer、ResNet 和 AdaLN-Zero 均为已有组件；这里的设计重点是逐动作区间条件化及其与轨迹复用的配合。

##### 4. 训练：学习不同起点到终点的平均流

Mean Flow 学习区间平均速度，而非 FM 的瞬时速度，因此可以用一次网络前向完成较长流区间的生成。

标准标量区间下，训练目标为：

\[
u_{\mathrm{tgt}}
=(A_t-e)-(r-s)\frac{d}{ds}u_\theta(A_t^s,O_t,s,r),
\]

\[
\mathcal L=
\mathbb E\left[
\left\|u_\theta-\operatorname{sg}(u_{\mathrm{tgt}})\right\|^2
\right].
\]

直观上，第一项提供从噪声指向示范动作的方向，导数项修正“瞬时速度”与“区间平均速度”的差异。导数使用 Jacobian-vector product 计算，目标端停止梯度。

AMF 将标量 \(s,r\) 扩展成向量 \(S,R\)，并混合两种训练扰动：

- **独立采样**：每个动作独立采样区间，增加噪声模式多样性；
- **单调采样**：按推理所需的时域结构采样 \(S\)，让训练分布更接近部署条件。

单调采样概率为 \(p_m=0.5\)。训练时扰动的基础是**真实示范动作**，不是模型上一步的预测；部署时才换成旧预测轨迹。

论文给出向量化训练伪代码，但未完整展开多索引情况下的导数定义与 JVP 方向；具体采样范围及部分分布参数也未给出。

##### 5. 推理：一次前向、执行一个动作、循环复用

每个控制周期计算：

\[
\hat A_t=A_t^S+(1-S)\odot
u_\theta(A_t^S,O_t,S,\mathbf1).
\]

只执行其中对应当前时刻的动作，然后将整段预测移位，供下一周期复用。首次预测没有旧轨迹，令 \(S=0\)，从纯噪声开始。

关键差异是：**每一步都完整去噪，再加入新噪声进行下一次重规划**，不是持续保留一条只被部分去噪的序列。

#### 方法对比分析

| 对比对象 | AMF 的本质差异 | 解决的痛点 |
|---|---|---|
| 普通 FM／MF 策略 | 不再每次从独立纯噪声生成，而是复用对齐后的旧预测 | 跨预测模式跳变与抖动 |
| MF-WarmStart | 不使用整段统一扰动，而是按时域分配扰动 | 避免稳定近期动作时也锁死远期计划 |
| Adaptive FM | 使用 Mean Flow 一步生成，而非多次流积分 | 降低闭环预测延迟 |
| Streaming Diffusion Policy | 每周期完整去噪后重新扰动，而非逐步部分去噪 | 作者假设新噪声带来的变化空间有助于适应 |
| Real-Time Chunking | 一致性主要来自输入初始化，而非去噪过程中的额外引导 | 避免额外测试时计算 |

创新主要是**一步 Mean Flow、分时域扰动、旧预测复用与匹配训练分布的组合机制**，而不是发明新的基础生成模型。

适合需要精确连续运动、频繁反馈，且示范存在多种解法的任务。对二值夹爪等离散行为，保守的轨迹继承可能反而妨碍及时切换。

#### 实验分析（精简版）

**验证协议。** 仿真覆盖 Push-T、动态目标块推送、D3IL 堆叠及 8 个 Metaworld 任务；比较分块／逐步 FM 和 MF、统一扰动复用、RTC、SDP 与 Consistency Policy。每次模型评估执行 50 次测试，汇总 3 个随机种子及最后 3 个检查点。真实实验包含 Push-T、合盖和传送带抓取。

**最有支撑力的两个结论：**

1. **收益不只是一步推理，而是生成速度与轨迹一致性的结合。**  
   仿真综合均分：AMF 为 **0.82**，Adaptive FM 为 **0.76**，统一扰动 MF-WarmStart 为 **0.66**，普通逐步 MF 为 **0.53**。Metaworld 动作三阶差分范数均值为 **0.98**，低于逐步 FM 的 **1.16** 和逐步 MF 的 **2.19**。但综合均分混合了成功率与重叠率，不应当解释为统一成功率；正文所称“6% 相对提升”也与表中四舍五入数字不完全一致，表面差值为 0.06。

2. **真实部署中，较低延迟伴随更好的任务表现。**  
   RTX 3090 上，AMF 延迟为 **14 ms**，FM-8 为 **54 ms**，FM-RTC 为 **79 ms**；真实 Push-T 成功率分别为 **0.60、0.35、0.35**。传送带平均成功率为 **0.55、0.48、0.40**。

消融支持使用逐 token AdaLN-Zero 和混合区间采样；仅使用单调采样时，Metaworld 平均成功率从 0.95 降至 0.77。

**证据边界：** 仿真未计入推理延迟；真实静态任务每方法每任务仅 20 次测试，传送带每速度仅 10 次，未报告置信区间。实验控制频率为 10 Hz，不能仅凭 14 ms 推理时间推断已验证更高频端到端控制。主要失败模式是任务末端停滞，尤其无法及时开合夹爪。

#### 实用指南

- **开放资源**：论文提供 [代码仓库链接](https://github.com/akselva/Adaptive-mean-flow-RoboticIL)。预训练权重、自采数据、许可证及仓库当前可用状态，论文未说明。
- **数据规模**：Metaworld 每任务 100 条示范；Push-T 90 条；Dyn-BP 1000 条；D3IL 部分报告 1000 episodes。真实机器人示范数量及明确的训练／验证划分未说明。
- **训练设置**：batch size 256；Push-T／Metaworld 500 epochs，Dyn-BP 200 epochs，D3IL 90 epochs；每 10 epochs 保存检查点。真实任务训练 600 epochs。学习率、优化器、训练硬件和训练耗时未说明。
- **实现优先级**：先核对噪声调度公式与代码，再确认逐动作 Mean Flow 的 JVP 实现、区间采样参数，以及动作序列移位和当前动作索引。损坏这些对齐关系会直接改变方法机制。
- **部署条件**：真实实验使用双 RealSense D435、UR10 或 Franka Panda，本地 RTX 3090 推理。实现需要支持自动微分 JVP 的框架；具体依赖版本和机器人软件栈未说明。
- **评估注意**：保持各方法观测时域、预测时域和网络容量一致，同时报告成功率、平滑性及实际闭环延迟。合盖任务采用抓取与放置各 0.5 分，不能直接把均分当作完整任务成功率。
- **迁移方式**：需要替换动作维度与单位、机器人状态编码、相机输入和对应示范数据，并重新训练或适配策略。更改控制频率后，应重新选择动作时域与扰动调度；夹爪等离散模态的独立处理属于作者提出的未来方向，尚未经验证。

#### 总结

核心思想：近端继承远端探索的一步重规划
1. 将上一周期预测移位对齐，保留包含历史动作的轨迹作为当前生成锚点。
2. 按预测时域扰动旧轨迹，让近期动作保持一致、远期动作获得重新规划空间。
3. 混合独立与单调区间训练，通过逐动作 AdaLN-Zero 学习不同扰动程度的一步平均流。
4. 结合最新观测一次生成完整轨迹，仅执行当前动作，再完整去噪、重新扰动并循环复用。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06089v1)
- [arXiv](https://arxiv.org/abs/2610.06089v1)

---

<a id='2610.06850v1'></a>
## [InterMimicGen: Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation](https://arxiv.org/abs/2610.06850v1)

**Authors:** Yucheng Zhang, Sirui Xu, Jinhong Li, Liuyu Bian, Anatulya Nandi, Derek Zhang, Xiangchen Liu, Xueting Li, Umar Iqbal, Yu-Xiong Wang, Liang-Yan Gui

**Published:** 2026-10-05

**Categories:** cs.RO, cs.CV, cs.GR

**Abstract:**

Captured human-object interactions provide rich supervision for humanoid loco-manipulation, but they are sparse, heterogeneous, and not directly executable by robots. We introduce InterMimicGen, a self-evolving motion-imitation framework in which robot motion data and a tracking policy improve each other. First, we consolidate motion-captured human-object interaction datasets and retarget them into humanoid robot references while preserving whole-body coordination and dexterous hand-object relationships. This produces a large and diverse humanoid robot reference collection for dexterous whole-body loco-manipulation. Second, we train a physics-based generalist tracker that executes these references in simulation on a humanoid with dexterous hands, covering a scale and diversity beyond prior humanoid tracking systems for loco-manipulation. Third, we close a data flywheel: each round makes small, task-preserving changes to where an interaction takes place and how the body performs it, fine-tunes the tracker on them, and keeps only the variants whose simulated execution completes the task, which seed the next round. With more iterations, these small edits compound into broader coverage around the sparse original demonstrations while preserving task semantics and motion quality. Experiments show contact-preserving retargeting across robot configurations, broad tracking with a single generalist policy, executable motions that keep growing over augmentation rounds, and transfer to real robots. InterMimicGen provides a unified path from heterogeneous human demonstrations to a continually expanding motion resource for humanoid robot learning.

### 论文解读

#### 摘要翻译

捕获的人–物交互为 humanoid 全身移动操作提供了丰富监督，但这类数据稀疏、异构，且不能直接由机器人执行。我们提出 InterMimicGen，一个自演化动作模仿框架，使机器人动作数据与跟踪策略相互改进。

首先，我们整合多种动作捕捉的人–物交互数据集，并将其重定向为 humanoid 机器人的参考动作，同时保留全身协调和灵巧手–物体关系。由此得到一个规模大、种类多的 humanoid 机器人参考动作集合，用于灵巧的全身移动操作。其次，我们训练一个基于物理仿真的通用跟踪器，使其在仿真中执行这些参考动作；其覆盖规模和多样性超过了以往面向移动操作的 humanoid 跟踪系统。第三，我们构建数据飞轮：每轮对交互发生的位置和身体动作方式作幅度较小、保持任务语义的修改，在修改后的动作上微调跟踪器，并只保留仿真执行成功的变体，作为下一轮的种子。随着迭代推进，这些小修改逐步扩展稀疏原始演示周围的覆盖范围，同时保持任务语义和动作质量。

实验显示，该方法能在不同机器人配置下进行保留接触关系的动作重定向；单个通用策略能覆盖多种动作；可执行动作会随增强轮次持续增长；生成动作也能迁移至真实机器人。InterMimicGen 为从异构的人类演示出发，持续扩展 humanoid 机器人学习所需的动作资源提供了一条统一路径。

#### 方法动机分析

人类演示记录的是高维交互空间中的少数可行轨迹，并非其所有可行变化。移动操作尤其脆弱：机器人必须在保持平衡的同时维持物体接触，几厘米的位置或姿态偏差就可能破坏抓取、造成穿透或导致摔倒。因此，仅让一个固定跟踪器执行固定参考动作，难以覆盖任务周围的可行变化；而单纯扩大固定参考集，也不能确保新增动作符合机器人动力学。

论文的核心假设是：**跟踪器可以修正经过小幅编辑但不完美的参考动作，而仿真执行成功的修正结果又能成为下一轮更远变化的训练数据。**因此，数据与策略应共同演化，而不是先固定数据、再训练策略。问题边界在于，系统扩展的是原始演示已经包含的任务语义，不是发明新任务；变体还必须保留交互类型、预期接触、物体和任务结果，并通过物理执行验证。

另一个设计动机是接触约束具有不同尺度：身体需要长期维持支撑和平衡及人与物的空间关系，手指则需要精细地围绕物体形成抓持。将两者压进单一优化目标，可能顾此失彼，因此论文采用先做全身交互重定向、再进行保接触的几何清理。

#### 方法设计详解

**输入与统一表示。**系统整合 InterAct 与 HiPHI 的交互数据，共 16,059 段动作、140.68 小时参考动作和 157 个数据集—物体条目。其中 9,030 段来自 InterAct，7,029 段来自 HiPHI。统一参考包含机器人状态、物体位姿与速度，以及接触意图；缺失的手部标注视为未知，而不是“没有接触”。每种机器人只保留其能力范围内的交互；例如，不支持灵巧手的机器人会排除依赖手指接触的任务。

**接触感知的动作重定向。**输入是人类身体、手部和物体轨迹，输出是目标机器人的关节动作及必要的物体位姿修正。

1. **全身交互求解：**以交互网格匹配身体关键点与采样物体几何之间的关系，而非只匹配关节姿态；在机器人关节、支撑和碰撞约束下优化浮动基座及身体。求解时固定原始物体轨迹，并用稀疏手部关键点确定手腕和手掌位置。
2. **手指对向约束：**对正在形成抓取的拇指—手指组合，惩罚它们从物体表面同一侧接近，促使手指从拇指对侧接触。直观上，这让手形成更有效的夹持，而不是仅仅贴近物体。
3. **几何清理：**两次 Adam 优化依次处理穿透与碰撞、轨迹平滑，并尽量保持原动作、足部支撑和手–物关系。每次清理结果都要通过偏差、穿透、接触保留和动作质量检查；不合格结果不接受。没有可动手指的机器人省略手指专用项。

**通用物理跟踪器。**策略输入机器人本体状态、未来参考帧、物体相对状态，以及身体各连杆到物体表面最近点的距离衰减几何特征；输出关节目标偏移，再由 PD 控制器产生力矩。策略以 PPO 在并行仿真环境中训练，奖励结合身体动作、物体运动、身体—物体交互关系和接触保真度。机器人与物体的跟踪项采用乘法组合，避免某一项表现好掩盖另一项严重失配。训练使用参考状态初始化和难例自适应采样；每个仿真环境只采样与其物体资产兼容的参考。

论文给出的通用策略训练设置包括：G1 配 Inspire 灵巧手、4 张 GPU、每张 GPU 4,096 个并行环境、50 Hz 控制频率和每个动作 4 个物理子步；PPO 每次采样 32 步，minibatch 为 16,384 个样本，训练 6 个 epoch。演员与评论家分别使用 1024→1024→512 的 ReLU MLP。其他 PPO 参数见论文附录；控制增益、动作尺度和力矩上限按机器人配置设定。

**自演化动作模仿。**每轮从已验证参考中选取父动作，提出两类变体：

- **物体编辑**改变交互发生的位置或方向，例如移动放置目标。编辑在预期接触阶段逐步加入，接触前保持原轨迹；机器人必须学习相应调整全身动作。
- **身体编辑**改变实现任务的姿势，例如骨盆、站姿、脚尖角度或肘部姿态，同时保持物体路径、接触意图及相应支撑约束。手臂编辑在保持手掌目标的空空间方向上改变肘部等姿态。

编辑后的候选动作重新计算运动学、速度和身体—物体特征。每轮先对当前已验证数据和候选动作微调跟踪器，再让跟踪器在物理仿真中执行候选动作。候选需要通过完成轨迹、不摔倒、动作平滑和任务结果检查；例如，放置任务需要物体稳定地停在支撑面上。每个候选执行 3 次，论文说明验证轮次会使用不同随机种子。成功执行的轨迹被存为新参考，供后续轮次继续编辑；失败变体不会进入下一轮。

这是论文的关键闭环：**编辑提出变化，策略通过模仿在动力学中修正变化，物理验证筛选可执行结果，筛选结果再扩充训练数据。**它不同于仅生成动作后过滤，也不同于只微调固定数据；策略学习能力的增长是持续接受更远变体的条件。

#### 方法对比分析

- **相较于固定参考动作的跟踪：**传统设置要求一个固定动作集合覆盖任务变化。本文把成功跟踪的变体回流为新参考，针对移动操作对微小姿态和物体位置变化敏感的问题，逐轮扩大可执行范围。
- **相较于只做场景缩放或物体位置变化的增强：**本文同时编辑交互位置和身体姿势，并保持父动作的任务语义与接触意图。身体编辑包括骨盆、站姿和手臂姿态等变化。
- **相较于单独的运动合成：**变体围绕已捕获的交互构造，再由物理跟踪策略执行和修正；并非依赖一个独立的运动合成器生成完整交互轨迹。论文认为，这能减少运动合成在物体交互中的脆弱性。
- **相较于单阶段重定向：**本文先用全身求解保留长期交互结构，再做局部几何清理，并用手指对向项改善抓持形状。创新点在于针对不同尺度拆解重定向目标，而不是只对身体关节轨迹作映射。
- **适用条件：**方法依赖有人–物交互参考、可仿真的物体与机器人模型，以及能检查任务结果的仿真环境；它扩展已存在的交互语义，不保证生成源数据之外的新任务。不同机器人需配置其身体关键点、手部映射和执行能力过滤。

#### 实验分析（精简版）

实验覆盖六种机器人配置：G1（含非灵巧手、Inspire 或 Dex3 手）、Booster T1、Booster K1 和 Dexmate Vega。重定向评估与 Weave 等方法比较；跟踪将单一通用策略与逐物体训练的专用策略比较；增强实验包含未增强策略和只筛选候选、但冻结策略的对照。论文还报告了多平台真实机器人执行。

最有力的两项结果是：

1. **增强数据和策略共同演化。**在 5 轮后，G1 + Inspire 的验证参考规模相对初始参考增长约 **146–151 倍**（抓取为 146.4 倍，双手交互为 150.5 倍）；演化后的策略对增强参考的成功率为 **98.4%–98.9%**，对原始参考为 **100%**。未增强策略在已接受变体上的成功率只有 **52.1%–64.2%**，说明成功变体不只是原策略已能执行动作的筛选结果。
2. **接触感知重定向改善动作质量。**在 G1 + Inspire 的 3,059 段 OMOMO 动作上，相较 Weave，本文方法的手接触帧比例为 **98.2% vs. 96.1%**，物体穿透帧比例为 **37.2% vs. 67.4%**，足滑帧比例为 **12.3% vs. 62.0%**，MPJPE 为 **5.74 cm vs. 9.54 cm**；但手内滑动略高，为 **0.239 m/s vs. 0.225 m/s**。

主要优势是大规模异构动作重定向、单策略跨物体跟踪，以及通过仿真闭环扩展已验证动作。局限包括：通用策略成功率仍低于逐物体专用策略，尤其是抓取任务；真实机器人实验展示了若干动作迁移，但不足以证明所有生成变体、所有任务或所有平台都可稳定部署。论文的覆盖范围也受种子动作、机器人能力过滤、仿真建模和任务验证规则限制。

#### 实用指南

- **代码、模型和数据：**论文给出了项目页面链接，但所提供内容未确认代码、模型权重或整合后的数据是否公开，也未给出可核实的下载许可；因此不能据此断言已经开源。
- **复现重点：**先统一身体—物体—手部表示并处理缺失接触标注；为目标机器人配置身体与指尖关键点、关节耦合和可支持任务；实现全身交互求解、手指对向和两阶段清理，再为每个物体准备仿真资产。之后按论文所述 PPO 配置训练跟踪器，并实现候选编辑、执行验证和成功参考回流。
- **评估注意事项：**区分跟踪成功率、物体误差、身体误差与轨迹质量；增强评估应同时报告演化前后策略在增强动作和原始动作上的成功率。论文的成功判断包括完成轨迹、摔倒、平滑度和任务结果，且每个候选有 3 次仿真执行；只报告保留数量会混淆策略学习与简单筛选的效果。
- **迁移到新机器人或任务：**需要替换或新增机器人关键点、手部映射、PD 控制参数、关节约束和能力过滤；手部结构变化还会影响抓取接触定义。若新增物体或任务，需准备相应资产和交互种子，并重新验证任务结果条件。论文未说明跨新任务自动获得任务语义或无需重新训练即可迁移。

#### 总结

核心思想：模仿与数据闭环共演化
1. 将异构人–物交互重定向为保留全身结构与手指接触的机器人参考。
2. 用通用物理跟踪器执行参考，并以交互几何和接触信息指导策略学习。
3. 对已验证动作作保持任务语义的物体或身体编辑，构造新变体。
4. 微调跟踪器后在物理仿真中验证候选，仅将成功执行的轨迹回流为下一轮种子。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.06850v1)
- [arXiv](https://arxiv.org/abs/2610.06850v1)

---

