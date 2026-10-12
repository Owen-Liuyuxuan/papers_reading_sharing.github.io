time: 20261012

# Arxiv Computer Vision Papers - 2026-10-12

## Executive Summary

## 今日研究速览

本期从100篇CS.CV/CS.RO记录中按关键词与领域适配规则保留27篇，逐篇评分后选出10篇并完成全文阅读。主题集中在具身策略的动作表示与运动预测、仿真数据生成和验证、主动视觉，以及面向未知动力学和约束流形的规划控制；自动驾驶方向则出现未来信息蒸馏与安全评分校正。

一个共同趋势是把“生成或预测”接入闭环检验：Grounded-Dreamer将合成操作轨迹送回模拟器验证，BOPI依据更新后的物理信念渐进修补正在执行的对象轨迹，FPD-Drive在训练时用未来观测筛选多样安全目标，再部署只看当前信息的学生模型。另一条路线通过更合适的中间表示改善策略学习：Human2Dex把人类手部语义与机器人执行适配分离，ReAT以真实解码残差逐token修正动作，Evo-World以未来光流潜变量补充VLA的运动线索。

值得优先精读的工作包括Human2Dex（两种真实灵巧手、六项任务，且明确区分共享数据接口和手型专属策略）、ReAT（多个模拟基准与一个真实堆叠任务共同检验动作token预算）、Grounded-Dreamer（比较合成轨迹与真实示范，并尝试OOD验证），以及BOPI（将信念变化直接映射为重规划幅度，并报告实体推车测试）。Evo-World的运动表征消融与真实双臂结果也值得关注。PATACON在高维约束规划基准展示了GPU加速，但没有实机验证；JoyInside的产品内成绩较好，不过公开基准表现不一，机器狗部分以定性案例为主。

整体证据仍以小规模实体试验和模拟器评估为主。解读这些结果时，应区分仿真成功率、真实机器人重复试验、定性演示和产品私有评估；多数方法的适用范围尚需更多任务、平台与失败条件验证。

---

## Table of Contents

1. [Human2Dex: Dexterous Robot Learning from Wrist-Centric Human Demonstrations](#2610.13119v1)
2. [Grounded-Dreamer: Robot Learning from World Model Synthetic Data with Physical Grounding](#2610.13164v1)
3. [Adaptive Whole-Body MPC with Optimal Posture Planning for Humanoid Robots under Unknown Payloads](#2610.12972v1)
4. [JoyInside-VLM-Agent: Product-Conditioned Multimodal Policy Learning for Embodied Agents](#2610.12957v1)
5. [ReAT: Residual Action Tokenization for Autoregressive Robot Policies](#2610.13093v1)
6. [Belief-Overlap Partial Inversion: Object-Centric Flow Replanning for Humanoid Loco-Manipulation](#2610.13087v1)
7. [EgoReveal: Learning Active Vision and Object Manipulation under Partial Occlusions from Egocentric Demonstrations](#2610.12999v1)
8. [Evo-World: A Lightweight Vision-Language-Action Model with Future Motion Prediction via Optical-Flow Supervision](#2610.12936v1)
9. [PATACON: GPU-Parallel Tangent-Bundle RRT-Connect for Constrained Motion Planning](#2610.13195v1)
10. [FPD-Drive: Future Policy Distillation for End-to-End Autonomous Driving](#2610.13232v1)

---

## Papers

<a id='2610.13119v1'></a>
## [Human2Dex: Dexterous Robot Learning from Wrist-Centric Human Demonstrations](https://arxiv.org/abs/2610.13119v1)

**Authors:** Jiachen Zhang, Yuanyuan Li, Shaqi Luo

**Published:** 2026-10-09

**Categories:** cs.RO

**Abstract:**

Collecting dexterous manipulation demonstrations for each robot hand requires repeated task-specific data acquisition. We ask whether one shared corpus of natural human demonstrations can instead serve as a reusable data interface for different dexterous hands, without collecting task demonstrations on each target robot. We present Human2Dex, a wrist-centric human demonstration interface that separates shared human observation and motion semantics from embodiment-specific execution. It aligns observations through local hand-appearance adaptation and a shared 21-point hand representation with consistent structural semantics across human and robot observations. It constructs executable supervision by retargeting fused human hand motion to each target hand, and introduces the Grasp-Pocket Interaction Representation (GPIR) to express tracked objects relative to a hand-centered Functional Grasp Pocket. The corpus is converted into separate training sets for Linker O6 and Wuji, while action labels, normalization, training, and diffusion-policy checkpoints remain hand-specific. Across six real-world tasks, Human2Dex achieves 109/120 (90.8%) successes on Linker O6 and 97/120 (80.8%) on Wuji. Across both hands, Human-Raw achieves 63/240 (26.3%) successes, whereas the complete Human2Dex configuration achieves 206/240 (85.8%). These results support the use of wrist-centric human demonstrations as a reusable interface across calibrated dexterous hands while retaining embodiment-specific policy learning. The project website is publicly available at https://human2dex.pages.dev/.

### 论文解读
#### 摘要翻译
Human2Dex研究如何让同一批自然人类示范服务于不同灵巧手，而不为每种机器人手重新采集任务示范。方法共享腕部视角观察与动作语义，再把人手运动重定向为各目标手专属的执行监督。作者在六项真实操作任务中测试Linker O6与Wuji，完整系统在两种手上都取得较高成功率，说明共享数据接口具有潜力，同时执行策略仍需按手型训练。

#### 方法动机分析
灵巧手数据采集昂贵，且人手与机器人手的外形、关节和可达动作不同，直接复用坐标会造成语义错位。论文的核心假设是：腕部观察和手部结构语义可跨手共享，但动作标签与策略必须适配目标手。这样既保留人类示范的规模，也避免假设不同执行器完全等价。

#### 方法设计详解
输入是腕部RGB观察、人手动作和目标手的结构信息。首先进行局部手部外观适配，并把人手和机器人手对齐到具有一致语义的21点表示；随后融合人手运动，通过目标手运动学重定向生成对应动作监督。物体状态另用Grasp-Pocket Interaction Representation表示：把物体相对功能抓握口袋定位，口袋由拇指与食指等关键点构成，并结合物体尺度、跟踪置信度和记忆状态。数据集由此转成各手独立训练集，最终分别训练扩散策略；共享的是观察接口和语义，动作归一化、训练过程与策略权重仍然按手区分。

#### 方法对比分析
与逐台机器人采集示范相比，它利用人类数据降低重复采集成本；与直接把手部坐标映射到机械手相比，显式的外观适配、结构语义和重定向更能处理形态差异。GPIR也比单一腕部坐标更关注物体和手的相对抓握关系。其创新在于把可共享的数据接口与不可忽视的执行差异放进同一流程，适合已有校准信息、希望复用人类示范的灵巧操作场景。

#### 实验分析（精简版）
作者采集1,899段、556,624帧的人类片段，并在两种机械手的六项真实任务上各评估120次。Linker O6成功109/120（90.8%），Wuji成功97/120（80.8%）；跨手合计完整配置206/240，Human-Raw基线为63/240。结果支持重定向接口有效，但完整系统同时加入跟踪、Pocket表示与辅助任务，无法从这组比较中分离各组件贡献；评测也只涵盖两种手与有限任务。

#### 实用指南
复现时需关注人手到目标手的结构标定、21点映射、物体尺度与跟踪质量，以及两类策略数据的独立归一化。项目网站可供进一步查阅，实际迁移应先验证新手型的可达性、接触几何和局部操作需求。若任务依赖按钮、把手或旋盖等局部接触，整体口袋表示可能需要补充手指级状态；论文结果不能直接推出免训练的跨手部署。

#### 总结
核心思想：共享语义，适配执行
1. 从腕部视频与人手运动提取统一手部语义。
2. 将物体相对抓握口袋表示并重定向到目标手。
3. 按手型生成监督并训练专属操作策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.13119v1)
- [arXiv](https://arxiv.org/abs/2610.13119v1)

---

<a id='2610.13164v1'></a>
## [Grounded-Dreamer: Robot Learning from World Model Synthetic Data with Physical Grounding](https://arxiv.org/abs/2610.13164v1)

**Authors:** Feng Jiang, Shu Chen, Yuanfei Wang, Yuan Li, Zhenhao Shen, Jiaqi Liang, Yang Chen, Lei Wei, Chen Xie, Ruihai Wu

**Published:** 2026-10-09

**Categories:** cs.RO

**Abstract:**

Scaling robot manipulation policies requires large amounts of diverse, action-labeled data, yet collecting such data in the real world remains expensive and difficult to scale. Existing methods either depend on human teleoperation, rely on simulation with limited visual realism, or use human videos that lack embodiment-specific annotations. In this paper, we introduce Grounded-Dreamer, a framework for synthesizing physically grounded robot manipulation data with video world models while avoiding human teleoperation demonstrations. Grounded-Dreamer combines the visual realism of generative video models with simulator physics engines. Starting from a small set of real images depicting robot task initial states, our method automatically reconstructs task-relevant objects and camera geometry, aligns them in simulation, and collects physically valid trajectories. These trajectories are used to post-train both a video world model and an inverse dynamics model, enabling the generation of visually realistic robot manipulation videos and corresponding robot actions from real images. To filter physically implausible rollouts, Grounded-Dreamer replays generated trajectories in the reconstructed simulation and retains only those that successfully complete the tasks. The real-world robot policies are trained solely on the grounded synthetic trajectories without any human demonstrations. Empirical results in real-world tasks demonstrate that Grounded-Dreamer can synthesize physically grounded data and substantially facilitate policy learning without requiring human teleoperation effort.

### 论文解读
#### 摘要翻译
Grounded-Dreamer希望从真实场景构造可以训练机器人操作策略的合成轨迹，减少昂贵的真实示范采集。它把场景重建、模拟器交互、视频生成和轨迹回放验证组成闭环，既生成多样视觉数据，也检查动作是否与模拟状态变化相容。论文在Franka/Panda平台的多类操作任务中测试，生成数据训练的策略接近真实示范基线，并在分布外任务设置中显示优势。

#### 方法动机分析
单靠生成视频扩充示范会产生接触关系不准、物体运动不符合动作等问题；单靠模拟器又可能缺少真实场景外观和足够多的交互轨迹。作者假设，真实场景重建能提供几何与外观基础，而模拟器中的交互、逆动力学检查和闭环回放能过滤掉不一致轨迹。视觉多样性与物理可执行性因此由不同环节负责。

#### 方法设计详解
流程先从真实环境重建场景和操作对象，在模拟器中配置任务并采样交互轨迹；对轨迹使用交互感知掩码突出手物接触区域，再由视频预测模型扩展动作过程。逆动力学模型将状态变化映射回动作，用于校验动作与接触是否协调。图像编辑模块生成不同的合成起始画面，增加外观变化。随后把候选轨迹放回模拟器闭环执行，依据动作和状态是否连续来筛选。生成策略训练数据后，在真实机器人上评估。主实验把只用合成轨迹训练的策略与每个任务使用50条真实演示训练的策略对比；该设置检验合成数据的独立价值，而非混合数据的总收益。

#### 方法对比分析
与开放式视频生成相比，Grounded-Dreamer要求候选轨迹能回到模拟器执行；与传统纯模拟数据相比，它从真实场景构造外观和对象条件，并借助视频模型扩展变化。创新重点是数据生成后的物理验证闭环。它适合已有场景重建和可交互模拟器、但缺少足量真实示范的操作任务；若仿真动力学偏差大，闭环通过也不必然意味着真实硬件可执行。

#### 实验分析（精简版）
论文在Franka/Panda上的七项真实任务中比较合成数据策略与真实示范策略。主合成方案仅用生成轨迹训练，整体接近每任务50条真实示范的基线，但各任务结果并不一致；分布外设置下合成数据更有优势。作者报告合成初始状态成功率约65%，表明扩增质量仍有限。实验展示了从模拟训练迁移到硬件的可能性，但任务数量、生成规模和物理环境覆盖尚不足以证明广泛泛化。

策略训练以生成轨迹为输入；部署推理仍需在目标机器人上检验闭环回放与延迟。

#### 实用指南
生成策略训练数据后，机器人部署推理仍需在目标硬件检验闭环成功率和耗时。
复现要点是场景和对象重建质量、任务轨迹采样、交互掩码、逆动力学校验以及回放筛选阈值；合成图像与物理轨迹应分别验收。文中每任务生成约100–150条交互轨迹，且多阶段模型与模拟器会带来明显资源成本。迁移到移动操作时应额外检查摩擦、遮挡和传感器误差；论文报告的硬件结果不等同于大规模部署验证。

#### 总结
核心思想：合成之后闭环验真
1. 重建真实场景并建立交互模拟任务。
2. 生成多样轨迹与图像，再做动力学检查。
3. 回放筛选数据，训练策略并上机评估。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.13164v1)
- [arXiv](https://arxiv.org/abs/2610.13164v1)

---

<a id='2610.12972v1'></a>
## [Adaptive Whole-Body MPC with Optimal Posture Planning for Humanoid Robots under Unknown Payloads](https://arxiv.org/abs/2610.12972v1)

**Authors:** Manh Cuong Nguyen, Duc Cuong Vu, Viet Anh Vu, Duc Tuan Pham, Toan Nguyen, Minh Hiep Phung, Ngo Anh Vien, Minh Nhat Vu

**Published:** 2026-10-09

**Categories:** cs.RO

**Abstract:**

Humanoid robots performing lifting and material-handling tasks must maintain balance despite unknown and varying payloads. However, payload uncertainty significantly affects whole-body dynamics and may destabilize conventional controllers. We propose an adaptive whole-body control architecture that integrates three components: (i) a centroidal model predictive controller (MPC) that generates dynamically feasible reference motions, (ii) an adaptive fuzzy balance controller that compensates for payload-induced model mismatch by generating corrective torques distributed across the supporting-leg joints, and (iii) a waypoint-based optimal posture planner built on a triple-integrator model that produces smooth sit-to-stand and stand-to-sit trajectories satisfying center-of-mass (CoM) stability constraints, joint limits, and whole-body feasibility requirements. Together, these components enable stable lowering, grasping, lifting, carrying, and placing behaviors under significant payload uncertainty. Extensive simulation and hardware experiments demonstrate that the proposed framework improves balance robustness and reduces tracking degradation during posture transitions and lifting tasks compared to non-adaptive baselines.

### 论文解读
#### 摘要翻译
本文研究人形机器人在不知道物体重量时如何稳定地抬起和搬运负载。作者提出自适应全身模型预测控制，将质心运动规划、姿态转换、平衡补偿和关节力矩执行分层组合，并在7kg与10kg载荷下进行仿真和硬件实验。结果展示了抬举及持物行走过程中的平衡能力，但实验规模和定量基线仍有限。

#### 方法动机分析
未知负载会改变机器人质心和躯干动力学，使预先规划的动作偏离稳定区域。单靠质心MPC未必能及时补偿实际跟踪误差，单纯增加关节优化又会带来更多约束与调参负担。作者的假设是：用较高层规划可行姿态，再由快速平衡反馈处理残余偏差，能够兼顾规划与扰动响应。

#### 方法设计详解
输入为机器人状态、接触信息和期望姿态。质心非线性MPC以50Hz生成全身运动参考；姿态规划器基于三重积分器求解满足关节位置、速度、加速度、jerk及质心支撑约束的姿态转换路径。逆动力学全身控制器跟踪加速度与接触力。自适应模糊平衡层以捕获点相对支撑多边形中心的偏差、质心速度和偏差积分为输入，通过规则系统估计俯仰与横滚补偿力矩，再由二次规划分配给支撑腿。模糊层和全身控制约以500Hz运行，底层关节控制超过1kHz。作者用逆动力学执行，而没有再增加一个任务空间二次规划层，以避免重复优化和任务权重调节。

#### 方法对比分析
与仅用质心MPC相比，新方法补上了负载引发的残余平衡误差反馈；与将所有任务并入大规模QP的做法相比，控制职责按层拆分。模糊规则可利用经验编码非线性误差响应，但其参数依赖专家经验，透明度和可迁移性有限。该设计更适合动力学模型不完备、又能提供质心与支撑信息的人形全身控制任务。

#### 实验分析（精简版）
作者用MuJoCo仿真，并在28自由度人形平台上测试7kg、10kg抬举及7kg负载抬起后行走。图示对照中，标准质心MPC在未知重载下失衡，完整方法保持了平衡并完成动作。现有证据主要是轨迹和关节时序曲线，没有大批次成功率统计；论文也指出尚未与PID做定量比较，因此改善幅度和跨平台稳定性仍需进一步验证。

#### 实用指南
实现时需要统一质心MPC、逆动力学控制和补偿力矩的坐标系与频率，并重新调节模糊隶属函数和力矩限制。姿态规划约束及负载范围应与硬件执行能力匹配。迁移到不同人形机器人时不能照搬经验规则，应补充负载、地面、速度和外部推扰测试，并报告成功率、恢复时间与能耗。

#### 总结
核心思想：规划姿态并快速补平衡
1. MPC生成可行的质心和姿态路径。
2. 模糊反馈从支撑误差估计补偿力矩。
3. 全身控制分配并执行关节动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.12972v1)
- [arXiv](https://arxiv.org/abs/2610.12972v1)

---

<a id='2610.12957v1'></a>
## [JoyInside-VLM-Agent: Product-Conditioned Multimodal Policy Learning for Embodied Agents](https://arxiv.org/abs/2610.12957v1)

**Authors:** Gangyi Li, LuYao Wang, Jingfu Sun, Jingyi Yuan, Hang Liu, Hao Zong, Yujia Hu, Tingting Gao, Jianyong Wang, Xu Yang

**Published:** 2026-10-09

**Categories:** cs.RO, cs.AI, cs.LG

**Abstract:**

Embodied agents must translate multimodal observations and user requests into executable actions while adapting to product constraints, changing goals, and execution feedback. We present JoyInside-VLM-Agent, a ReAct-style multimodal agent developed through product-conditioned trajectory construction and staged policy learning for robotic platforms and smart hardware. A closed-loop sandbox generates multimodal interaction trajectories covering goal revisions and failure recovery, which are checked against execution outcomes and semantic criteria. Supervised fine-tuning and multi-teacher on-policy distillation precede reinforcement learning with DPER (Dual-Source Privileged On-Policy Self-Distillation with Token-Level Evidence Routing). DPER routes token-level evidence between a frozen privileged teacher and a privileged student sharing the rollout policy's parameters, modulating reward-derived token advantages while preserving their signs. Training rewards combine generative process evaluation, multi-judge rankings, and task-queue consistency. JoyInside-VLM-Agent achieves a BFCL v4 overall score of 77.44, exceeding the evaluated baselines; results on the other public benchmarks are mixed. On JoyInside-Devices and JoyInside-Robotics, the evaluated system, which includes domain-specific skill refinement, leads the baselines on all reported metrics. Two real-world quadruped case studies illustrate visually grounded search, replanning after execution failure, and explicit failure reporting.

### 论文解读
#### 摘要翻译
JoyInside-VLM-Agent面向带有产品规格、可用工具与交互记忆的多模态具身代理。论文先构造产品条件化的可执行交互数据，再通过监督微调、多教师在线蒸馏和奖励优化训练单一学生模型。公开工具使用基准呈现有胜有负，产品内部评估较强；真实机器狗部分主要是案例展示，尚未提供大规模硬件成功率。

产品代理的痛点还包括执行历史与当前目标不一致，论文以保持产品约束为训练驱动力。

#### 方法动机分析
其痛点是静态数据难覆盖产品约束、目标修订与失败恢复，这正是分阶段交互学习的驱动力。
通用视觉语言模型可能会调用不存在的工具、违反设备约束，或在用户修改目标和操作失败后无法恢复。作者认为，仅用静态问答数据难以训练持续交互能力；此外，轨迹级奖励能评价任务结果，却难以说明奖励应该影响哪些动作token。因此需要可执行的产品环境和细粒度的训练证据路由。

这一训练pipeline明确区分数据执行、教师指导和学生部署三个阶段。

#### 方法设计详解
完整pipeline依次是沙盒轨迹构造、分阶段训练和学生策略推理部署。
数据构造阶段把产品角色、工作流、约束、工具与技能装入沙盒，按用户图像和目标生成闭环轨迹，再做执行与语义验证，覆盖目标修订和失败恢复。策略训练先以多来源示范进行监督微调；随后学生在自身生成的前缀上接受语言教师和视觉教师的在线策略蒸馏；最后DPER根据token位置选择教师或学生证据，结合生成过程奖励模型、多评审排序和任务队列一致性，重标定优势。奖励缩放尽量保持原奖励符号，减少排序信号反转。系统从Qwen3.5-35B-A3B初始化，约120K训练示例和约50K私有交互。教师仅参与训练，部署时只运行学生策略。

#### 方法对比分析
与只做监督微调相比，在线蒸馏可以在学生实际生成的前缀上吸收语言和视觉教师行为；与单一终局奖励相比，DPER试图把不同来源的证据分配给对应token。创新在于产品条件化数据与分阶段训练的整体设计。它适合工具、产品状态和操作规则明确的服务型代理，但系统收益难以从复杂训练配方中拆分归因。

公开评估中BFCL v4分数77.44，对照大模型为72.90。

#### 实验分析（精简版）
BFCL v4上77.44分，相较基线72.90分有所提高。
在BFCL v4上，系统得分77.44，高于Qwen3.5-397B-A17B的72.90；在MSAgent上得分90.39，低于该大模型的92.41；ToolACE轨迹准确率51.53也略低于52.27。产品内Devices与Robotics结果领先，但包括领域技能精炼。实体机器狗展示导航、视觉报告和对话案例，没有足够的重复试验统计，故公开基准与真实机器人证据应分别解读。

#### 实用指南
复现依赖可靠的产品规格、可执行沙盒、轨迹验证和各类奖励信号，不能只复制模型训练阶段名称。需分别报告通用基准、产品技能和硬件试验，并对多教师、DPER及奖励组件做消融。迁移到机器人时，可先用于工具规划和失败恢复；安全的实体动作还需独立的约束检查与量化验证。

对阅读者而言，理解奖励如何落到具体token，是区分方法贡献与数据规模效应的关键。
#### 总结
核心思想：产品约束贯穿交互学习
1. 将产品规则和工具装入可执行沙盒。
2. 生成并验证闭环多模态交互轨迹。
3. 通过多教师和奖励路由训练部署学生。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.12957v1)
- [arXiv](https://arxiv.org/abs/2610.12957v1)

---

<a id='2610.13093v1'></a>
## [ReAT: Residual Action Tokenization for Autoregressive Robot Policies](https://arxiv.org/abs/2610.13093v1)

**Authors:** Dijia Zhan, Zhuo Li, Xuemiao Xu, Sun Zheng, Jinyi Li, Wenjun Wang, Jie Tang

**Published:** 2026-10-09

**Categories:** cs.RO

**Abstract:**

Autoregressive robot policies rely on action tokenizers to convert continuous action chunks into compact sequences of discrete prediction targets. Existing tokenizers can reconstruct action chunks from discrete codes, but often leave implicit what each individual token contributes to the reconstructed action. We introduce Residual Action Tokenization (ReAT), which gives each token an explicit role in refining the current action reconstruction. It uses the decoded reconstruction as feedback to construct the next token from the remaining action error. This closes the loop between encoding and decoding, making each new token responsive to the errors left by the actual decoded prefix. Across 19 manipulation tasks spanning three simulation benchmarks and real-world settings, 8-token ReAT policies outperform the evaluated baselines in task success; on LIBERO, even 4-token ReAT surpasses the evaluated 8-token action-tokenization baselines at half the token budget. These results suggest a path toward action tokenization that supports self-corrective policy learning, helping autoregressive policies learn to progressively refine their own action plans.

### 论文解读
#### 摘要翻译
ReAT提出带重建反馈的动作token化方法。每个新token不独立表示整段动作，而是描述当前解码草稿尚未重建的残差；decoder更新动作后，再以新的重建误差构造下一token。该策略旨在让有限token预算更有效地表达连续动作。论文在多个模拟操作基准和真实机械臂桶堆叠任务中验证，并分析了动作成功率与推理延迟之间的权衡。

#### 方法动机分析
动作离散化能让策略像语言模型一样逐token预测，但固定码本或独立token可能忽略此前已解码动作的误差，短序列下尤其容易损失精度。作者假设，若每一步都根据当前解码结果重新计算残差，后续token便能针对性修正累积误差，而不是重复描述已表达的内容。

#### 方法设计详解
将专家动作块记为目标，从零动作草稿开始。第k步计算目标与前k−1个token解码结果的差值，将该残差送入Transformer残差编码器；有限标量量化把编码映射为离散索引和嵌入。冻结decoder结合已解码草稿与新嵌入，产生修正后的动作，再从这个真实解码结果计算下一残差。编码器在每个阶段复用，但读取对应阶段的寄存器特征；策略训练时对每种前缀长度施加监督。动作使用累积平移坐标，降低不同参考点带来的误差。部署端策略从视觉观察预测token，再调用decoder逐步恢复动作块。

与OAT和QueST相比，ReAT在八token配置的多数基准上更优。

#### 方法对比分析
相比OAT与QueST，ReAT在八token多数基准有优势，但并非各预算和延迟指标都占优。
相较于一次性离散化，ReAT的token有明确的顺序修正作用；相较于传统自回归动作序列，它把少量token用于逐步逼近整个动作块。关键差别是残差针对实际解码结果，而非潜在特征差异。适合动作预测预算受限、又能接受逐步解码的机器人策略；若极端强调最低延迟，则需要根据token数和decoder成本实测。

在真实任务中，八token ReAT达到70%成功率，而OAT为40%。

#### 实验分析（精简版）
真实堆叠成功率方面，ReAT八token为70%，OAT八token为40%。
八token设置下，LIBERO成功率71.16%、MetaWorld 31.70%、RoboCasa 55.90%，分别比同预算下最强基线高约3.28、1.80和2.00个百分点。真实桶堆叠每配置20次，ReAT八token成功14次，OAT八token成功8次，QueST八token成功7次。RTX 6000D、batch size 1的八token端到端延迟约22.2–25.4ms，二token为7.6–11.0ms，说明性能与速度取决于预算，不是统一加速。

#### 实用指南
复现需保持残差坐标、冻结解码器和前缀监督一致，并在相同硬件、batch size与动作时域下比较延迟。真实实验使用90段示范，但只有一个任务且每配置20次。迁移到不同机械臂或动作空间时，应单独确认动作归一化及坐标定义；不能仅凭重建误差判断控制效果。

作者还显示动作重建误差与策略成功并非一一对应，因此必须同时报告任务指标和推理代价。
部署端仍需按真实机械臂的动作时域和安全限制重新选择token预算。
#### 总结
核心思想：每个token修正真实残差
1. 从零草稿开始重建动作块。
2. 量化目标与当前解码结果的残差。
3. 用下一个token更新草稿并继续修正。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.13093v1)
- [arXiv](https://arxiv.org/abs/2610.13093v1)

---

<a id='2610.13087v1'></a>
## [Belief-Overlap Partial Inversion: Object-Centric Flow Replanning for Humanoid Loco-Manipulation](https://arxiv.org/abs/2610.13087v1)

**Authors:** Chen Yu, Sudarshan S Harithas, Juan Borbon, Dingqi Zhang, Jiuguang Wang

**Published:** 2026-10-09

**Categories:** cs.RO

**Abstract:**

People often learn how an unfamiliar object moves through a small exploratory push, using its response to guide their next motion. Inspired by this behavior, we study humanoids pushing carts whose mass and caster stiffness are initially unknown. Our object-centric framework closes online estimation, flow-based planning, and whole-body control into a loop, creating a tension: replacing the plan whenever the belief shifts disrupts the maneuver under way, while keeping it strands the robot on a path its evidence has ruled out. Belief-Overlap Partial Inversion (BOPI) scales revision to agreement, inverting the flow under the old belief only as far as the beliefs disagree before reconstructing under the new one. Across 18 simulated goals it lowers pooled final position and yaw error by roughly 33% and 47% against fixed-depth replanning, and on hardware it translates and turns a non-holonomic cart.

### 论文解读
#### 摘要翻译
BOPI研究移动操作中物体属性逐渐被估计后，机器人如何更新正在执行的对象轨迹。它依据旧、新物理信念下生成轨迹的分布重叠程度，决定回退已有流场轨迹多远，再使用更新后的信念重建剩余计划。仿真和购物车硬件实验显示，该策略可降低目标误差并减少不必要的路径变化。

#### 方法动机分析
推车质量、脚轮约束等动力学参数起初未知，交互过程中估计更新后，原计划可能不再合适。完全保留旧计划会忽视新信息，完全重采样又会突然改变正在执行的动作。作者的核心假设是：新旧信念越相似，计划越应保持；差异越大，才需要重写更多轨迹。

#### 方法设计详解
系统以目标和交互历史为输入，估计器更新物体质量、后脚轮约束等信念，再由对象级流规划器向全身控制器提供轨迹。BOPI在旧信念下生成的轨迹上反向积分，反向深度由新旧轨迹分布的Bhattacharyya重叠决定；随后在新信念下正向积分，得到修改后的未来路径。信念接近时几乎保留原路线，分歧较大时则重新规划更长片段。控制器持续跟踪对象航点，形成估计、计划、控制闭环。理论分析说明在Lipschitz等假设下，信念重叠趋近完全一致时，编码计划的变化趋近零。

相较固定深度重采样，本文的创新是按新旧信念相似度调整计划修补范围。

#### 方法对比分析
相较固定深度的方案，创新在于依信念重叠自适应地选择回退与重建范围。
RDM固定加入一定噪声，信念改变大小不影响重规划深度；FlowEdit依赖新旧流场差值。BOPI根据分布重叠自适应反演程度，重点在稳定地修补当前对象计划。它适合执行期间物体属性逐渐可知、且保持轨迹连续很重要的移动操作；理论界依赖其动力学与正则性假设，不构成普遍安全保证。

#### 实验分析（精简版）
G1-Dex3仿真覆盖18个目标和每目标32个随机种子，BOPI汇总位置/朝向误差为0.179米/0.152弧度，RDM为0.267米/0.286弧度。购物车硬件平移与转向各进行5次，BOPI平均终点位置误差约5.78厘米，RDM为8.59厘米。硬件任务数量很少，结果尚不能说明复杂场景泛化。

#### 实用指南
实现需统一旧新条件下的轨迹潜表示，并保证信念重叠能反映真实动力学变化。硬件重规划按1Hz执行以顾及安全；估计误差突变、障碍约束和执行延迟都应单独验证。可迁移到移动底盘推物，但需补测多种物体、推力和地面条件。

该设置让结果更能检验在线修补，但目标数与每种方法试验数仍有限。
算法收益应结合硬件控制频率和状态估计质量一起评估。
#### 总结
核心思想：信念相近就少改计划
1. 从交互历史估计物体动力学。
2. 用新旧信念的轨迹重叠确定回退深度。
3. 在更新信念下重建并执行剩余路径。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.13087v1)
- [arXiv](https://arxiv.org/abs/2610.13087v1)

---

<a id='2610.12999v1'></a>
## [EgoReveal: Learning Active Vision and Object Manipulation under Partial Occlusions from Egocentric Demonstrations](https://arxiv.org/abs/2610.12999v1)

**Authors:** Hiromu Kurihara, Kazutoshi Tanaka, Masashi Hatano, Tadashi Kozuno, Hideo Saito, Mariko Isogawa

**Published:** 2026-10-09

**Categories:** cs.RO

**Abstract:**

Learning from Demonstrations (LfD) enables robots to learn object manipulation directly from human demonstrations. In contrast to resource-intensive and labor-intensive teleoperation, egocentric demonstrations are easy to collect, enabling scalable data collection. Manipulated objects are often partially occluded in everyday environments, yet prior egocentric demonstration-based methods that require no robots or specialized teaching devices do not explicitly address occlusions. To fill this gap, we propose EgoReveal, a framework that learns viewpoint-repositioning together with manipulation actions from egocentric human demonstrations to reposition the viewpoint and obtain target visibility under partial occlusions. EgoReveal instantiates this as two imitation policies: one predicts the future camera-arm pose sequence from the current view and camera-arm pose, and the other predicts the future control commands from object keypoints and a history of control commands. EgoReveal switches between the two based on target visibility. In experiments where the manipulated object is partially occluded initially, EgoReveal repositions the camera to obtain visibility and completes the task more reliably than fixed-camera baselines.

### 论文解读
#### 摘要翻译
EgoReveal通过主动移动相机来处理机器人抓取中的部分遮挡。系统先控制相机手臂寻找更清晰的目标视角，再在遮挡被判断为解除后切换到独立的操纵策略。论文使用智能眼镜采集第一视角人类示范，在玩具和毛巾任务上进行测试，结果优于固定视角基线；测试范围仍较窄。

#### 方法动机分析
固定相机下，遮挡物可能挡住目标，导致视觉特征或抓取关键点落在错误位置。直接要求操纵策略在遮挡下完成定位，会混合视角规划与抓取控制两个问题。作者假设先通过主动视觉恢复可见性，再切换到常规操作策略，可以让两类能力分别学习和评估。

#### 方法设计详解
智能眼镜采集第一视角视频，相机图像先做鱼眼矫正，并与相机臂位姿对齐。Active Vision策略采用ACT，根据当前图像预测一段未来相机位姿，使相机绕开遮挡。遮挡评估模块ORA先用OwlViT检测候选目标，再按尺寸和图像语义相似度筛选；SAM3分割目标区域，并比较参考物体与当前区域的HSV颜色直方图。随后跟踪掩码面积在一段时间内的变化率，若面积趋稳，判定目标已充分显露并切换策略。操纵阶段由EgoZero根据RGB、深度和关键点估计抓取动作。切换时提取的目标关键点在后续操纵中使用。

#### 方法对比分析
与固定相机的EgoZero相比，EgoReveal增加了主动视角策略和显式可见性切换；与预设固定等待时间相比，ORA根据检测、分割和面积变化判断何时交接。模块化便于替换操纵器，但当前关键点在切换后保持固定，若再次遮挡或定位漂移，系统缺少重新观察步骤。适合部分遮挡且相机手臂可移动的桌面操作。

#### 实验分析（精简版）
Soft Toy和Towel任务成功率分别85%和90%，未见遮挡条件为70%；不使用主动视觉时对应35%、45%、30%。去掉ORA的结果为55%、70%、50%，显示主动观察和切换规则都有作用。论文没有报告表中各条件的试验次数，且只涵盖固定底座、并行夹爪和部分遮挡，证据边界较窄。

#### 实用指南
复现需校准相机位姿、鱼眼矫正、检测分割阈值及切换时的关键点坐标。迁移时应覆盖光照、目标外观、再次遮挡和多目标干扰，并明确记录各条件试验次数。更通用的操纵策略可替换EgoZero，但主动观察策略仍需与新相机运动学适配。

这类分阶段系统尤其需要报告交接错误，因为单项策略成功并不保证连续执行可靠。
尤其应验证遮挡重新出现时系统能否安全停止或恢复观察。
#### 总结
核心思想：先主动看清再抓取
1. 从第一视角观察并预测相机运动。
2. 检测、分割目标并确认可见性稳定。
3. 切换到操纵策略完成抓取。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.12999v1)
- [arXiv](https://arxiv.org/abs/2610.12999v1)

---

<a id='2610.12936v1'></a>
## [Evo-World: A Lightweight Vision-Language-Action Model with Future Motion Prediction via Optical-Flow Supervision](https://arxiv.org/abs/2610.12936v1)

**Authors:** Zile Zhong, Yuxin Du, Tao Lin, Haoxiao Wang, Wei Dai, Binhua Zuo, Bo Zhao

**Published:** 2026-10-09

**Categories:** cs.RO

**Abstract:**

Vision-language-action (VLA) models provide a unified framework for visual understanding, language grounding, and robot control, yet anticipating how the scene will evolve beyond the current observation can provide additional information for control. Action supervision alone does not explicitly encourage such anticipation, while future-image prediction entangles task-relevant motion with largely static appearance. To address these limitations, we introduce Evo-World, a lightweight VLA model that augments policy learning with future motion prediction through optical-flow supervision, thereby providing a task-relevant predictive signal focused on motion rather than appearance. We further design the training procedure to preserve these predictive representations during policy learning, regulating representation diversity while decoupling representation learning from action optimization. Across simulation benchmarks, Evo-World demonstrates strong performance, achieving a tier-averaged success rate of 91.1% on Meta-World MT50 , with a 13.0% relative improvement over the non-predictive baseline. On real-world AgileX robots, Evo-World outperforms the evaluated baselines across single-arm and bimanual manipulation tasks while using only 0.87B parameters and delivering lower GPU memory usage and higher inference efficiency. These results demonstrate that future motion prediction through optical-flow supervision provides an effective and lightweight approach to enhancing the predictive capabilities of VLA models.

### 论文解读
#### 摘要翻译
Evo-World为视觉语言动作策略加入一个未来运动潜变量预测器，使策略除了观察当前图像和语言指令，也能利用场景将如何变化的表征。训练时从未来光流构造监督，推理时不需要未来图像或光流。论文在模拟操作基准和AgileX双臂机器人任务上评估，显示运动预测能提升策略表现，同时模型规模和显存需求较小。

#### 方法动机分析
只根据当前画面生成动作，可能难以判断动作执行后物体会如何移动。预测未来RGB既复杂又容易受外观变化干扰。作者假设光流更直接反映场景位移，把未来运动压缩成潜在特征后调制视觉语言动作策略，能够为动作提供动态线索而不需要显式重建未来图像。

#### 方法设计详解
输入包括指定视角图像和文本指令。DINOv2将当前图像编码成视觉特征，CLIP编码指令，运动预测器据此预测未来运动潜变量。训练时用RAFT计算15帧之后的光流，将光流渲染为三通道图像并通过DINOv2得到监督目标；该未来目标仅参与训练。随后将当前与预测运动表征拼接，经停止梯度处理后送入FiLM模块，生成缩放和平移参数来调制InternVL3-1B视觉语言特征。经调制的上下文条件化flow-matching动作专家，输出连续动作。FiLM末层零初始化以保留初始策略，SIGReg约束表示，梯度截断减少动作损失反向影响运动表征。部署无需未来光流。

#### 方法对比分析
与不预测未来的VLA相比，Evo-World明确把运动变化作为条件；与未来RGB预测相比，它使用光流潜表征而非像素重建。消融结果支持光流目标和FiLM融合的价值。方法适合操作中对象运动对后续动作重要的任务，但2D位移表征不等于三维几何或物理状态建模。

#### 实验分析（精简版）
MetaWorld MT50 tier平均成功率91.1%，无未来运动预测基线为80.6%。AgileX双臂上插花、挂杯和放入微波炉各20次，成功率55%、70%、65%，平均63.3%，对照π0为51.7%。实机仅覆盖三项任务且每项使用100条遥操作示范微调。模型0.87B参数、显存2.54GB，推理约6.16Hz，结果依赖报告的硬件和训练设置。

#### 实用指南
复现需要未来光流数据、RAFT预处理、DINOv2目标编码以及视觉语言策略微调；训练与推理数据路径需区分，部署端不应假设能读取未来观测。迁移到新机器人时应验证视角选择、光流质量和动作频率。若任务需要深度、接触力或遮挡后的三维状态，需增加额外表征与评测。

因此迁移到高精度接触任务时，最好补充深度或力觉信号，而不是只依赖二维位移特征。
#### 总结
核心思想：用未来运动调制当前策略
1. 从当前画面与指令预测未来运动特征。
2. 训练时以未来光流潜变量监督预测。
3. 用FiLM融合运动与视觉语言特征生成动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.12936v1)
- [arXiv](https://arxiv.org/abs/2610.12936v1)

---

<a id='2610.13195v1'></a>
## [PATACON: GPU-Parallel Tangent-Bundle RRT-Connect for Constrained Motion Planning](https://arxiv.org/abs/2610.13195v1)

**Authors:** Gahyun Oh, Junho An, Suhyun Jeon, Suhan Park

**Published:** 2026-10-09

**Categories:** cs.RO

**Abstract:**

Fast whole-body motion planning enables continuous replanning for humanoid robots in dynamic environments while satisfying collision avoidance and kinematic constraints. However, constrained motion planning incurs substantial computational cost due to repeated constraint projection in addition to collision checking and nearest-neighbor search. Recently, GPU-based massively parallel planners have been proposed to reduce computational cost. Nevertheless, these planners still steer in the ambient configuration space, so projecting candidates that deviate from the manifold remains a significant source of overhead, particularly in highly constrained, high-dimensional problems. In this work, we introduce PATACON, a GPU-parallel bidirectional planner that explores tangent space to exploit the local manifold geometry induced by known constraint functions. Experiments on dual-arm closed-chain and humanoid whole-body constrained planning tasks show that PATACON outperforms the prior work as configuration-space dimension and constraint codimension grow, achieving up to an 11.25 times speedup. These results suggest that GPU-parallel tangent-space exploration brings constrained whole-body motion planning closer to real-time performance.

### 论文解读
#### 摘要翻译
PATACON提出面向约束流形机器人规划的GPU并行RRT-Connect算法。它在局部切空间中引导采样，以多候选边扩展和并行验证提高搜索效率，同时保留约束投影和可行路径检查。作者在不同自由度的规划基准上测试，结果显示高维约束任务可获得速度收益；低维简单问题则未必更快，且没有实体机器人验证。

#### 方法动机分析
机器人关节和接触约束会把可行状态限制在低维流形上。传统采样器可能在不可行区域浪费探索时间，路径逐段投影与碰撞检查也难以充分利用GPU并行。作者假设沿局部切空间采样、一次提出多个边候选并批量验证，可减少无效扩展并摊薄投影计算成本。

#### 方法设计详解
算法先为树节点建立切空间基，并以TGS根据局部几何选定扩展树和切空间邻域。随后TGMP沿同一切向方向生成多个连续候选节点；每个候选航点仍需投影回约束流形，投影步骤按序进行。边检查则分配给GPU线程组并行处理，若某段失败，算法将最长的可行前缀加入树中。双向树交替扩展并通过固定目标进展测试完成CONNECT。另有回退抑制策略，避免在新切空间中朝父节点方向重复探索。算法以GPU全局内存维护起点和终点树的切空间信息。

#### 方法对比分析
相较普通RRT-Connect，PATACON把采样限制到局部切空间并利用GPU并行候选验证；相比只做GPU批量采样，它还显式管理切空间并复用多步扩展。优势主要出现在高维或约束密集的问题；额外的切空间构造与投影开销会让低维简单任务不占优。其结果应按问题维度、约束类型和硬件分别判断。

具体而言，G1基准PATACON平均152.95毫秒，cpRRTC平均242.48毫秒。

#### 实验分析（精简版）
G1平均规划时间为152.95毫秒，对比cpRRTC的242.48毫秒。
在G1的20维约束基准上，每类随机问题100次，PATACON平均规划时间152.95±44.29ms，cpRRTC为242.48±19.47ms，两者成功率均100%。Franka低维基准中cpRRTC为0.52ms，PATACON为1.65ms，显示低维下反而较慢。实验在RTX 5090工作站进行，未报告实体机器人运行，因此属于规划基准证据。

#### 实用指南
复现需实现约束雅可比与切空间基、稳定的流形投影、GPU并行边检查，并按统一终止条件统计成功试验时间。注意论文修订表格更正过早期对照数据，应采用最终表中数值。迁移到真实机器人前需把碰撞检测、控制周期和规划失败恢复纳入闭环测试。

实际吞吐仍会受到流形投影串行部分、GPU调度和具体约束雅可比实现影响。
因此它更像高维规划加速组件，而非单独的机器人控制方案。
#### 总结
核心思想：沿约束切空间并行扩展
1. 构造可行状态附近的切空间。
2. 批量生成候选边并并行验证。
3. 将最长有效路径前缀并入搜索树。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.13195v1)
- [arXiv](https://arxiv.org/abs/2610.13195v1)

---

<a id='2610.13232v1'></a>
## [FPD-Drive: Future Policy Distillation for End-to-End Autonomous Driving](https://arxiv.org/abs/2610.13232v1)

**Authors:** Li-Heng Chen, Qifeng Chen, Chengwei Wu, Yingjie Zhang, Sheng Yang, Hongbo Fu, Shaoqing Ren

**Published:** 2026-10-09

**Categories:** cs.CV

**Abstract:**

End-to-end autonomous driving models are typically trained on real-world driving logs, which record only the trajectory actually executed in each scene. This provides limited supervision for learning alternative valid behaviors. Multi-target supervision can mitigate this limitation, but broader trajectory coverage makes safe candidate selection more challenging. We propose FPD-Drive, an end-to-end planner that distills privileged future knowledge into a visual driving policy to improve both multi-trajectory supervision and safety selection. Specifically, we introduce Future-Privileged Multi-Target Distillation to transfer diverse, high-quality trajectory targets from a teacher with access to recorded future observations to the student. These targets extend supervision beyond the single recorded human trajectory. For candidate selection, we further introduce Safety-Aware Score Correction to correct ranking errors between nearby safe and unsafe candidates and raise underestimated safety scores of safe candidates. At deployment, the student requires neither future observations nor teacher inference. Our base model achieves 94.9 PDMS on NAVSIM v1 navtest and 55.9 EPDMS on NAVSIM v2 navhard. The scaled FPD-Drive-Pro improves these scores to 95.5 PDMS and 58.1 EPDMS, respectively, achieving SOTA performance among the compared methods on both benchmarks. Zero-shot closed-loop evaluation on HUGSIM further yields an HD-Score of 45.1 and RC of 55.3.

### 论文解读
#### 摘要翻译
FPD-Drive面向自动驾驶轨迹规划，采用训练时可访问未来观测的教师模型帮助当前传感器学生策略学习。教师从多条候选中挑选安全且多样的轨迹，并用安全感知评分校正减少评分器偏差；部署时仅运行学生。NAVSIM基准和HUGSIM闭环仿真显示性能提升，但论文尚未进行真实车辆测试。

#### 方法动机分析
单一记录轨迹无法覆盖相同场景下的多种安全路线，而学习到的轨迹评分又可能把危险候选排得过高，或低估合理安全路线。作者的假设是，训练时利用未来观测生成更优且多样的目标，再显式修正安全与评分之间的不一致，可以让只依赖当前观测的学生获得更可靠排序。

#### 方法设计详解
输入包括四路当前相机、车辆状态和导航指令，学生先生成多条粗轨迹，再经扩散细化器修正并由评分器排序。教师在训练时还读取记录的未来观测和全局场景特征，据此细化并评价候选。未来特权多目标蒸馏从教师候选中选择最多四条既安全又有几何差异的轨迹，用最优匹配监督学生提案和细化结果。安全感知评分校正检查候选之间的关系：若高分轨迹实际不安全，则相对邻近安全候选降低其优势；若安全候选评分偏低，则提高其logit。训练后部署仅保留学生。扩散细化器使用8步DDIM推理。

#### 方法对比分析
相较只模仿单条记录轨迹，教师提供多个经过筛选的目标；相较直接拟合评分器，评分校正引入安全候选间的相对约束。关键贡献是把未来特权用于训练监督，再把安全排序修正注入学生。它适用于离线数据包含未来状态、而部署只能使用当前传感器的轨迹规划，但仿真安全分数不能替代真实道路验证。

NAVSIM v1的PDMS分数为94.9%，Pro为95.5%；NAVSIM v2的EPDMS为55.9和58.1。

#### 实验分析（精简版）
NAVSIM v1 PDMS得分94.9%，Pro为95.5%；NAVSIM v2 EPDMS为55.9和58.1。HUGSIM场景RC均值55.3，HD-Score为45.1。
NAVSIM v1上，基础版PDMS为94.9，Pro为95.5；NAVSIM v2 EPDMS分别55.9和58.1。HUGSIM初步闭环评估包含345个场景，平均RC为55.3、HD-Score为45.1。Pro额外使用158.8K合成场景，因此不能把全部提升单独归因于评分校正。所有结果来自基准或仿真，论文没有真实车辆部署证据。

#### 实用指南
复现需要当前与未来数据对齐、教师候选筛选、轨迹安全评估及学生蒸馏，并保持NAVSIM和HUGSIM指标定义一致。训练使用8张H800，推理含8步扩散细化，部署成本需据目标硬件重新测量。迁移到机器人导航时，可借鉴训练特权与部署输入分离，但需额外定义碰撞风险、动态障碍和闭环安全验证。

应重点检查教师选出的目标是否满足场景安全约束，而不能只验证其与记录轨迹的几何相似度。
#### 总结
核心思想：用未来教师教当前学生选安全路线
1. 生成并细化多条候选轨迹。
2. 教师筛选多样安全目标并校正评分。
3. 蒸馏到只看当前观测的部署学生。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.13232v1)
- [arXiv](https://arxiv.org/abs/2610.13232v1)

---

