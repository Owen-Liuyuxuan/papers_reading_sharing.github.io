time: 20261029

# Arxiv Computer Vision Papers - 2026-10-29

## Executive Summary

## 本期概览
本期从完整获取的188篇计算机视觉与机器人论文中纳入188篇候选，按直接用途、创新性、研究广度与证据强度排序，最终精读10篇。主题横跨机器人操作、跨域数据采集、水下定位、闭环自动驾驶规划，以及统一视觉生成与图像先验。

最值得关注的一条系统方向，是让模型的中间决策更贴近真实执行：RoboAuto把场景量测、物理知识和程序验证接进机器人搜索；ExecDiffuser则指出驾驶扩散模型的关键不只是生成候选，而是发布并执行哪一条。两项工作都强调环境反馈和逐步核验，不过目前实机证据仍有限，ExecDiffuser仅有仿真评测。

另一组论文关注视觉表示与数据质量。OpenJaws用双目定位改善操作示教轨迹，并直接测试定位误差对策略的影响；SO-101 Bench覆盖空间关系、鲁棒性和遮挡记忆，同时发现仿真成功率会高估实机。SVIn+展示水压、DVL与声呐如何补足水下视觉惯性估计。对于统一生成，Know, then Draw把语义知识获取交给自回归预训练、把像素渲染交给扩散；NLD-Uni则以跨范式蒸馏和反复细化增强离散扩散多模态模型。Image Modeling综述提供了从手工先验到基础图像模型的贝叶斯视角。

若时间有限，建议先读RoboAuto和ExecDiffuser，比较两种“生成/搜索—验证/发布—反馈”设计；随后读SO-101 Bench与OpenJaws，了解基准和数据质量如何影响具身策略。关注视觉表示和生成的读者可接着阅读Know, then Draw、NLD-Uni及Image Modeling。上述结果来自不同任务和评测协议，跨论文分数不可直接横比。

---

## Table of Contents

1. [SO-101 Bench: Accessible Spatial Manipulation](#2610.36420v1)
2. [RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding](#2610.36533v1)
3. [OpenJaws: An Open Hardware UMI Framework with Stereo Localization for Generalist Manipulation](#2610.36598v1)
4. [ExecDiffuser: Closing the Sample-to-Execution Gap in Diffusion Planning](#2610.36837v1)
5. [Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well](#2610.37652v1)
6. [ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning](#2610.37666v1)
7. [Nemotron-Labs-Diffusion-Uni: Elastic Self-Refining Unified Diffusion LLM with Cross-Paradigm On-Policy Distillation](#2610.36466v1)
8. [Know, then Draw: A New Learning Paradigm for Visual Generation](#2610.37626v1)
9. [Image Modeling: From Handcrafted Priors to Foundation Image Models](#2610.36646v1)
10. [SVIn+: Multi-Modal Framework for Robust Underwater State Estimation using 3D Sonar, Visual-Inertial, Water-Pressure, and DVL](#2610.36454v1)

---

## Papers

<a id='2610.36420v1'></a>
## [SO-101 Bench: Accessible Spatial Manipulation](https://arxiv.org/abs/2610.36420v1)

**Authors:** Truman Hickok

**Published:** 2026-10-27

**Categories:** cs.CV

**Abstract:**

Low-cost arms such as the SO-101 have become the most common entry point to robot learning, yet no released policy performs spatially constrained, language-conditioned tasks on them out of the box, and existing SO-101 benchmarks run only on physical hardware with simple tasks and few held-out objects. We introduce SO-101 Bench, a simulated benchmark for language-conditioned spatial manipulation built on an Isaac Lab digital twin of a real SO-101 setup. It has four tasks scored as five families, 138 objects in seen, unseen-instance and unseen-class splits, 916 evaluation episodes with subsets for held-out appearances, rotate-to-fit planning and occlusion memory, and 3,629 simulated demonstrations on which every learned policy is fine-tuned under an equal compute budget. The twin ranks policies as the real robot does: five policies trained without real data, four VLAs and an LLM agent, keep their exact ranking when the same weights run on the physical arm in an unseen environment with unseen objects (Spearman rho = 1.00, MMRV = 0.000), although simulation overestimates success. Across nine VLAs and world-action models, the best learned policies score near 61%, with spatial placement, unseen object classes and rotate-to-fit planning remaining open. An LLM agent that writes code and hands grasping to a learned policy scores 83.2%, and an ablation shows that this lead comes from the choice of harness and the choice of language model.

### 论文解读
#### 摘要翻译
SO-101成本低、易获得，却缺少评估空间语言操作的标准基准。作者构建Isaac Lab数字孪生，设计空间关系、物体扰动、旋转放置和遮挡记忆任务，并检验仿真结果能否预测真机表现。结果显示仿真适合比较策略，但成功率会系统性高估真实效果。
#### 方法动机分析
许多桌面操作基准集中于抓取或单步放置，难以测“放在A与B之间”“先转向再放入”等空间约束，也较少评估遮挡后的目标记忆。作者的假设是：围绕常见低成本机械臂构建覆盖多种空间语义的可复现任务，并用实机校验仿真，可以让策略比较更可操作。
#### 方法设计详解
基准包含138件家庭物体和916个episode，划分已见物体、未见实例、未见类别与私有测试集。任务覆盖单/多容器放置、相邻关系、两物之间移动及空间规划；ROBUST考察扰动，PLAN考察旋转后放置，MEM考察遮挡记忆。作者在仿真中提供3,629条演示，再以一致的36小时真机数据微调策略。真机比较也改变控制接口，分别评估直接动作、代码策略与委托抓取等方式。 输入是自然语言指令、当前图像和机械臂状态，输出是可执行的抓取与放置动作；数字孪生自动记录物体类别、目标关系和成功条件。分层测试使失败可归因于物体识别、空间规划、记忆或接触控制，而不是只得到一个总分。 基准输入为语言指令和环境状态，输出是自动评分的任务完成结果；对象划分可检查模型是否只记住训练物体。

#### 方法对比分析
它的贡献是评测覆盖和sim-to-real分析，而非新策略结构。与只看抓取成功率的基准相比，它能区分语义理解、几何约束、鲁棒性和记忆；但策略排序稳定不代表仿真数值能替代真机测试。
#### 实验分析（精简版）
五种仿真策略在有限跨域实验中的排名相关系数为1，但仿真成功率最高高估真机31%。实机仅45个episode、一个未见环境，证据支持“仿真可初筛”，不支持普遍预测能力。 这说明基准的主要证据是相对排序与绝对差距同时存在；实机样本很少，尚需更多操作者、布局和机械臂个体重复验证。

#### 实用指南
适合低成本桌面操作研究者复用任务划分、隐藏物体测试和真机校验流程。部署时应控制夹爪、代码执行方式和抓取模块等因素；论文报告每个策略36小时真机微调。需要额外核对仿真物理参数与真实接触差异。
#### 总结
核心思想：用真机校准空间操作基准
1. 在数字孪生中定义空间、扰动与记忆任务。
2. 训练和统一微调多类策略。
3. 用真实机器人核对排序与绝对成功率。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36420v1)
- [arXiv](https://arxiv.org/abs/2610.36420v1)

---

<a id='2610.36533v1'></a>
## [RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding](https://arxiv.org/abs/2610.36533v1)

**Authors:** Buzhao Liu, Jianing Ye, Xiu Yuan, Yangtong Feng, Chengsong Huang, Tyler Dinh, Yevgeniy Vorobeychik, Ning Zhang, Chongjie Zhang

**Published:** 2026-10-27

**Categories:** cs.RO, cs.AI, cs.CL, cs.CV, cs.LG

**Abstract:**

Recently, LLM-based autonomous agents have shown strong capabilities in iteratively generating, evaluating, and refining executable solutions. In robotic manipulation, Code-as-Policy lets these agents search over modular and editable robot programs. However, effective program search requires physical understanding: agents must understand how objects can be manipulated, how scene geometry constrains actions, and why interactions succeed or fail. Existing methods rely mainly on execution feedback for program refinement, often leaving these physical questions unresolved. We introduce RoboAuto, a framework for robot program search that explicitly organizes physical understanding at the object, scene, and interaction levels through three components. First, a hierarchical knowledge base combines executable skills with object-centric knowledge of affordances, spatial relations, and action effects, learned from embodied data and online task solving. Second, scene grounding connects this knowledge to the current environment by identifying relevant objects and resolving action-critical quantities. Third, multi-stage planning with self-designed experiments checks whether each stage produces its expected physical effects and actively tests competing hypotheses to resolve physical uncertainties. RoboAuto achieves superior performance across five benchmark families, attaining an 89.1% success rate on short-horizon LIBERO-PRO tasks and more than twice the mean success rate of ASPIRE on LIBERO-PRO Long and LIBERO-Plus. It also outperforms ASPIRE and $π_{0.5}$ on three challenging long-horizon real-robot tasks requiring precise manipulation. We further show that the knowledge base transfers across simulators and embodiments, and that program search performance scales with its size. Project page: https://roboauto-ai.github.io/

### 论文解读
#### 摘要翻译
RoboAuto通过多层物理知识理解来搜索机器人程序。系统从对象affordance、空间关系和动作效果中积累知识，以当前场景量测实例化动作，再分阶段搜索、执行并验证程序；通过实验区分物理假设，将成功程序沉淀为技能。
#### 方法动机分析
代码策略可根据执行结果修订程序，但常重复犯错：把抓取点放在不合适位置、把一个物体的高度套到另一个物体，或未确认动作效果就继续。作者认为，单靠代码搜索不足以弥补缺少物理知识，因此需要把对象知识、场景几何和执行反馈串联。
#### 方法设计详解
系统先维护包含对象affordance、空间关系、动作效果和已验证程序的层级知识库。收到指令后，视觉工具定位目标和支撑物，测量动作相关距离、角度等量；规划器把任务拆成有前置/后置状态的阶段，执行后由验证器检查状态变化。若出现失败，审计器识别可能错误的假设，并设计能区分解释的实验；新证据更新知识库，可靠程序保存为可复用技能。系统使用共享的感知、指向、分割、抓取与控制原语。 输入是语言任务、图像和执行历史，输出是带阶段状态条件的机器人程序及其验证证据。知识库给出可查询事实，grounding模块提供当前场景几何量，搜索器负责产生动作，验证器决定是否继续；失败实验的结果再回写知识库。

#### 方法对比分析
相较仅靠执行轨迹修补程序的代码策略，RoboAuto显式记录可迁移的物理事实，并在当前场景重新grounding。核心优势是可解释和复用，代价是依赖工具精度、编码代理及知识库正确性。
#### 实验分析（精简版）
LIBERO-PRO空间扰动成功率94.8%，ASPIRE为55.3%；LIBERO-Plus分别90.3%与44.7%。RoboTwin加入知识库后91.5%，移除后24.5%。真实机器人只涉及三个任务，仍需人类复位和安全介入，证据不足以证明真实环境全自治。 多基准结果提示它在组合步骤和场景变化上受益明显，但真实测试没有覆盖长时运行、人员安全或未经整理的家庭环境。 因此更适合把分项性能与人工干预记录一起阅读。

#### 实用指南
适合研究可执行代码策略、长期任务和错误恢复的团队。初始化知识库需要演示，复现还依赖Claude Opus 4.6及论文中的感知控制primitive；部署前应逐步验证检测、定位、接触和状态判定可靠性。 对需要长程操作的应用，逐阶段检查能让开发者更快定位物理假设与代码错误。

#### 总结
核心思想：让机器人从验证过的经验中搜索
1. 从指令提取目标并测量场景。
2. 分阶段生成、执行和验证程序。
3. 用区分性实验修正知识并沉淀技能。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36533v1)
- [arXiv](https://arxiv.org/abs/2610.36533v1)

---

<a id='2610.36598v1'></a>
## [OpenJaws: An Open Hardware UMI Framework with Stereo Localization for Generalist Manipulation](https://arxiv.org/abs/2610.36598v1)

**Authors:** Adithyavairavan Murali, Yash Oza, Karanbir Chahal, Neel Jawale, Silas De Vries, Hesam Rabeti, Erwin Coumans, Balakumar Sundaralingam, Stan Birchfield

**Published:** 2026-10-27

**Categories:** cs.RO

**Abstract:**

Handheld data-collection interfaces such as the Universal Manipulation Interface (UMI) allow manipulation demonstrations to be gathered in the wild without a robot in the loop, but the quality of the resulting action labels is bounded by the accuracy of vision-based localization. Existing systems recover end-effector pose from monocular visual-inertial odometry, which is subject to scale ambiguity and drift, or delegate tracking to closed-source modules and external infrastructure that reintroduce cost and setup burden. We present OpenJaws, a fully open hardware and software framework for sensorized UMI that improves localization and simplifies deployment across robot platforms. OpenJaws pairs a modular, 3D-printed gripper exposing a standardized ISO adapter with a low-cost stereo camera to recover metric camera trajectories. We further introduce UMI-Loc-Bench-v1, a benchmark quantifying real-world localization accuracy for handheld data-collection devices, and show that stereo localization substantially outperforms monocular alternatives. Finally, we demonstrate that policies trained on data collected with OpenJaws can be deployed on a real robot across multiple tasks. Hardware, software, and data are released under a commercially permissive license. Code, data, and videos are available at https://open-jaws.github.io/

### 论文解读
#### 摘要翻译
OpenJaws开源一套低成本立体UMI采集与机器人执行框架，包含双目手持示教器、柔性夹爪、定位基准和仿真资源。作者通过定位评测、真实策略部署和真实到仿真重建展示系统的实用性。
#### 方法动机分析
单目手持相机缺少绝对尺度，SLAM误差会污染示教动作标签；不同采集设备和机器人夹爪之间还要适配坐标与几何。作者用双目视觉直接恢复米制轨迹，并将采集器和执行夹爪设计成可替换、开放的模块。 论文的驱动力是降低示教设备成本和跨机器人集成挑战，同时避免定位误差长期混入动作标签。

#### 方法设计详解
硬件采用全局快门双目相机与3D打印TPU柔性两指夹爪，接口遵循ISO 9409-1-50-4-M6标准。UMI-Loc-Bench-v1收集9类操作、每类10段手持示范；ArUco标记提供评估真值，但被定位算法屏蔽。作者比较COLMAP、cuVSLAM、ORB-SLAM3、UMI-SLAM和ViPE，并把定位误差传递到UR10上的堆叠、抓放策略，另尝试重建轨迹并在数字孪生回放。 输入是双目图像和手持末端运动，输出为米制相机/末端轨迹及可用于策略学习的数据。双目特征匹配恢复尺度，SLAM或SfM模块估计位姿，再由统一坐标变换映射至机器人TCP；策略部署阶段用标准夹爪接口执行采集的动作片段。

#### 方法对比分析
双目提供无需额外标记的尺度，较单目方法更适合动作标签；COLMAP精度高但耗时，cuVSLAM在速度和精度间折中。工作重点是整合硬件、定位与数据集，而非提出新SLAM算法。 相比闭源或依赖头显追踪的方案，OpenJaws使用开放SLAM框架和低价立体相机；COLMAP偏准确但离线耗时，cuVSLAM适合较快处理。贡献是整套数据采集链路的开放化。

#### 实验分析（精简版）
完整轨迹COLMAP为5.2毫米/0.8度，cuVSLAM为12.4毫米/1.9度，运行时间分别836秒和34.8秒。真实堆叠/抓放成功约90%；注入定位漂移后成功率降至30%。实机与仿真验证仅覆盖有限夹爪、平台和任务。 注入漂移是较直接的因果压力测试，但属于合成扰动，不能覆盖自然手持运动、遮挡和夹爪磨损的全部误差分布。

#### 实用指南
适合希望快速搭建UMI数据采集、评测定位质量和复现真实到仿真流程的实验室。部署要校准双目与机器人TCP，并记录完整轨迹覆盖率；低成本BOM约399美元，定位模块可按离线精度或在线吞吐选择。 若选择离线COLMAP应预留计算时间，在线采集则可优先考察cuVSLAM的吞吐。 复现者应保存相机到末端的刚性变换，避免采集和部署坐标不一致。

#### 总结
核心思想：用立体定位守住操作数据质量
1. 双目设备采集带尺度的操作演示。
2. 比较定位方法并生成动作标签。
3. 在实体机器人与仿真中检验策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36598v1)
- [arXiv](https://arxiv.org/abs/2610.36598v1)

---

<a id='2610.36837v1'></a>
## [ExecDiffuser: Closing the Sample-to-Execution Gap in Diffusion Planning](https://arxiv.org/abs/2610.36837v1)

**Authors:** Shuo Pei, Bosun Liang, Saihui Zha, Chen Sun, Huachun Tan, Yong Wang

**Published:** 2026-10-27

**Categories:** cs.RO

**Abstract:**

Generative planners are often evaluated by the trajectories they can generate, although closed-loop driving ultimately depends on the single candidate committed to control. We identify this mismatch as the sample-to-execution gap and show that improving proposal quality alone can worsen the executed decision. Here, we introduce ExecDiffuser, a recurrent proposal-to-execution framework that couples candidate generation, trajectory selection, and subsequent replanning through shared execution semantics. Intent-Guided Diffusion Proposal (IGD) assigns hypothesis responsibility and shapes trajectories with execution-conditioned supervision. Execution-Grounded Selection (EGS) learns a candidate-conditioned publication utility, commits one trajectory to control, and returns it as context for the next planning cycle. Across nuPlan, InterPlan, and NAVSIM, ExecDiffuser achieves leading closed-loop performance against strong state-of-the-art baselines. Under a unified five-seed protocol, it obtains an average score of 85.62, improving over Diffusion Planner and FlowDrive by 4.02 and 1.31 points, respectively. On NAVSIM, it achieves 92.6 PDMS under the official evaluation pipeline. Matched counterfactual analysis further shows that ExecDiffuser more reliably converts strong candidates into committed decisions, while prefix interventions confirm that this advantage persists after physical execution and subsequent replanning.

### 论文解读
#### 摘要翻译
ExecDiffuser指出扩散规划生成了多个候选，但真实闭环只会执行其中一条。方法将意图引导扩散生成与执行证据驱动的候选发布分开，使已提交轨迹及其后续状态反馈到下一规划周期。仿真结果显示这一设计能改善闭环得分和交互场景恢复。
#### 方法动机分析
很多工作用候选oracle衡量规划器，忽略选择器最终发布什么、该动作又怎样改变下一时刻交通状态。作者的核心假设是：规划器应同时学习生成有效候选和判断哪条在当前执行上下文中值得发布，并将已执行决策纳入递归上下文。
#### 方法设计详解
场景编码器读取地图、路线、历史和上一发布轨迹。IGD根据潜在意图查询生成轨迹，并以几何约束塑形；候选数在2、4、8间自适应。EGS检查共享证据向量和硬有效性，再按效用选出一条轨迹，明显无效时拒绝；若全无有效候选则触发确定性最小风险机动。提交轨迹未执行部分会在下一周期对齐后重新使用，但模型不跨周期反传梯度。配置观察2秒、规划8秒、10Hz、25步扩散推理。 输入为场景观测、地图路线和上一发布计划，输出为单条闭环执行轨迹。IGD负责提出意图，EGS从共享证据中判断有效性与效用，递归接口把上次承诺的未执行后缀对齐到下一周期；这将候选生成、筛选和反馈形成闭环流程。

#### 方法对比分析
相较只优化生成分布或事后用规则选候选，ExecDiffuser联合考虑候选意图、有效性、风险和连续性。它直接优化“发布决策”，但发布层的有效性取决于预测质量和代理指标，不能提供形式化安全保证。 与只看候选oracle或用固定规则排序不同，方法把候选提交后的状态变化纳入分析，并用匹配前缀实验隔离初始发布的影响；适用于交互强、计划承诺会改变后续环境的场景。

#### 实验分析（精简版）
nuPlan五种子平均85.62，比Diffusion Planner高4.02、FlowDrive高1.31；NAVSIM PDMS为92.6。4秒固定前缀后，EGS比模仿选择的恢复分高1.90。全部是仿真结果；遮挡路口false-valid为5.14%，且推理延迟59.5毫秒。 NAVSIM五个独立种子的PDMS均值为92.6，标准差0.29；这一结果支持仿真跨基准稳定性，但不能代替真实道路安全验证。

#### 实用指南
适合扩散驾驶规划及任何“生成多个候选但只提交一个”的闭环系统。实现时要记录执行后状态、验证候选有效性并报告全链路延迟；论文使用25步解码和五种子重训。机器人迁移是方法启发，需重新定义安全约束和动作执行反馈。
#### 总结
核心思想：按执行后果发布规划
1. 生成并塑形多个意图候选。
2. 按证据筛除并发布一条。
3. 将已发布计划反馈到下一周期。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36837v1)
- [arXiv](https://arxiv.org/abs/2610.36837v1)

---

<a id='2610.37652v1'></a>
## [Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well](https://arxiv.org/abs/2610.37652v1)

**Authors:** Ian Chuang, Hiya Shah, Vamsi Chilamkur, Iman Soltani, Claire Tomlin

**Published:** 2026-10-27

**Categories:** cs.RO, cs.LG, eess.SY

**Abstract:**

Dexterous manipulation with multi-fingered hands is the next frontier of robotics, yet learned hand skills remain limited to simple grasps or basic repeated motions, such as rotating an object. As robots attempt more demanding in-hand maneuvers, safety, such as not dropping the object, becomes critical, yet it is rarely measured. We study in-hand object pose reaching, a task that demands diverse and difficult grasps: moving an object to almost any position and orientation within reach of the fingers, at any wrist orientation, through long sequences of consecutive goals. We present a recipe for training such policies for a LEAP hand in simulation without human data, built on a range of insights, from diverse resets and goals via generated grasp datasets to an adaptive friction curriculum. With careful system identification, the policies transfer zero-shot to the real hand, using either motion capture or a single wrist-mounted depth camera, and prove remarkably capable and robust. On top of the task policy, we add a learned residual safety filter with adjustable conservativeness, letting us tune the balance between performance and safety and greatly reduce object drops. In 13.3 hours of real-robot evaluation over 4,800 unseen goals across 4 objects and diverse wrist orientations, the state-based policy reaches 95.2% of goals within 2 cm and 0.4 rad, and the filter cuts its drops by 75% while still reaching 91.5% of goals. The policies can also spin a lightbulb held in an arbitrary pose, reach out-of-distribution goals, withstand fast arm motion and perturbations, and, needing only a wrist camera, serve as portable skills such as screwing in a lightbulb. We release the full recipe as a platform for studying performance, reliability, and safety in high-dimensional dexterous systems. Videos and more are available at https://ian-chuang.github.io/dexsafe/.

### 论文解读

#### 摘要翻译
本文研究多指灵巧手能否把物体连续移动到手掌坐标系的不同位置和姿态，同时减少掉落。作者在仿真中训练 LEAP Hand 位姿到达策略，并在其上叠加可调的残差安全过滤器。真实机器人实验覆盖四种物体、六种腕部朝向和数千个连续目标；过滤器明显减少掉落，但会牺牲部分目标成功率。作者还展示了仅用腕部深度相机进行旋转和组合操作的可能性。

#### 方法动机分析
既有灵巧操作常集中在单轴旋转、固定腕部方向或短序列。真实手内操作还要求物体平移到指尖可支撑的位置，并在长时间运行中控制掉落。作者假设从可抓取姿态构造目标、逐步降低仿真摩擦，能让策略学习更广的操控；单独训练安全过滤器则可调节安全与成功率之间的取舍。

#### 方法设计详解
作者为每种物体生成抓取库：先用 LightningGrasp 产生十万抓取，再用 RRT 扩展到五十万，碰撞过滤后保留约 42–45 万。初始状态和连续目标均从库中抽取。仿真采用经过实机辨识的伺服器模型和延迟，并随机化摩擦、质量、几何、扰动及深度噪声；物理仿真 120 Hz，策略控制 20 Hz。

训练先用 PPO 和特权状态学习教师，再以在线 DAgger 蒸馏为实机可用的状态策略，之后 PPO 微调；最后训练读取腕部 160×120 深度图的视觉策略，并加入辅助位姿预测。课程从高摩擦和较近目标逐步转向现实摩擦及全抓取库目标。任务奖励衡量位置、朝向、转速和保持时间的进展，并惩罚掉落、自碰撞、非指尖接触与剧烈动作。安全过滤器在冻结策略动作上加残差，以存活奖励减去加权修正代价训练；参数 w 控制干预强度，状态过滤器随后被蒸馏为深度输入版本。

#### 方法对比分析
与仅靠任务奖励间接降低掉落相比，独立过滤器可单独调节干预，无需重训任务策略。与基于安全价值函数的过滤器相比，它只需一次网络前向计算，不必在线搜索候选动作；代价是没有形式化安全保证，也可能压制有用动作。视觉策略还把感知误差带入控制闭环，不能与状态策略混为一谈。

#### 实验分析（精简版）
主实验含 4,800 个连续目标。状态策略在 2 cm、0.4 rad 阈值下成功率为 95.2%；过滤后为 91.5%，但掉落从 1,200 个目标中的 16 次降至 4 次。深度策略过滤后成功率从 71.8% 降到 66.7%，掉落由 88 次降至 46 次。视觉位姿误差在刀具和酒杯上有长尾；脚本化翻转、拧灯泡等任务只做有限试验，不能证明自主高层规划。

#### 实用指南
论文称将发布代码和检查点。复现时需重点校准伺服动力学、延迟、碰撞模型和低摩擦范围。抓取库使目标更可达，但不保证每个姿态真实可行。策略与过滤器都按物体训练；迁移到新物体或手型需重新建库、辨识和训练，不能直接当作通用安全层。

#### 总结
核心思想：分开学习灵巧操作与安全修正
1. 从抓取姿态构造多样目标。
2. PPO 学任务，再蒸馏为状态和深度策略。
3. 残差过滤器修正动作，调节安全与成功率。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.37652v1)
- [arXiv](https://arxiv.org/abs/2610.37652v1)

---

<a id='2610.37666v1'></a>
## [ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning](https://arxiv.org/abs/2610.37666v1)

**Authors:** Chaehyun Kim, Chi-Pin Huang, Jisoo Kim, Min-Hung Chen, Yu-Chiang Frank Wang, Seungryong Kim, Fu-En Yang

**Published:** 2026-10-27

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Vision-Language-Action (VLA) models have made substantial progress in robotic manipulation, but their generalization remains constrained by the diversity and coverage of action-labeled robot demonstrations. Human videos provide much broader behavioral and environmental experience, while learning from them faces a fundamental trade-off between scalable data and explicit action grounding. We introduce ThinkWorld, a VLA learning framework that grounds action reasoning in world dynamics and strengthens it through reinforcement learning with world imagination. ThinkWorld represents each action thought as a Chain-of-Dynamics (CoD), a sequence of discrete motion tokens that captures the intermediate dynamics connecting the current world to a future state and serves as a common action representation across embodiments. We first initialize CoD with action-labeled demonstrations, then predict the future induced by each CoD and reinforce those whose predictions better match the observed outcome. This enables videos without action annotations to provide direct supervision from observed state transitions, allowing human and robot experience to refine the same reasoning space. Experiments in simulation and on real robots demonstrate effective human-to-robot transfer and strong generalization to unseen tasks and environments.

### 论文解读

#### 摘要翻译
ThinkWorld 希望利用数量丰富的人类操作视频改善机器人策略泛化。它将人类与机器人动作编码成共享的离散运动序列，再学习这些运动会如何改变场景。训练时，模型预测的未来视觉特征与视频中实际出现的后续状态越一致，对应动作推理就得到越高奖励。因此，没有动作标签的视频也能参与第二阶段训练。方法在模拟基准和双臂机器人任务中展示了跨物体、轨迹和环境变化的迁移能力。

#### 方法动机分析
机器人示范包含可执行动作，却受限于实体平台、采集成本和单一机械臂的控制接口；人类视频覆盖的动作与环境更丰富，但不能直接转成机器人关节命令。常见潜在动作可能把摄像机运动或外观变化也当作动作。作者的假设是：如果动作表征能预测它造成的后续世界变化，就能用视频结果约束动作含义，并在不同身体形态之间形成可迁移的中间表示。

#### 方法设计详解
输入包括任务指令、当前图像、历史画面和机器人状态。人类手腕轨迹与机器人末端运动先转成位姿增量，再由运动 tokenizer 编成共享的 CoD（Chain-of-Dynamics）离散 token。视觉语言模型学习根据上下文预测 CoD。未来状态分支在 CoD 后加入可学习查询，经解码器预测约一秒后的 DINOv3 图像 patch 特征；训练只需比较预测特征与真实后续帧特征，不需要生成像素视频。

第一阶段用带动作标签的人类和机器人示范训练 CoD，并同步训练未来特征预测。第二阶段从相同上下文采样多个 CoD，依据预测未来与实际未来的相似度计算奖励，再用 GRPO 优化模型；少量监督回放保留动作 token 的落地能力。奖励会提高物体或手部等动态区域的重要性，同时仍保留全局场景一致性。连续机器人动作由单独的 action expert 产生，CoD 不是直接下发的控制命令。未来预测和 CoD 推理只参与训练，实际部署时不运行这两个组件。

#### 方法对比分析
与仅用动作标签训练相比，ThinkWorld的创新是要求候选运动能解释观测到的后果；它也不同于只从人类视频抽取未约束潜动作；不同于把未来视频作为控制输出，它把预测一致性用作训练奖励。模型仍需带标签的机器人动作建立连续控制能力，目标机器人也要经过具身专属微调，所以人类视频扩展了训练监督，却没有消除机器人示范需求。

#### 实验分析（精简版）
LIBERO 平均成功率为 97.3%，LIBERO-PRO 为 55.6%。在 LIBERO-PRO 的任务变化和环境变化条件下，成功率分别为 31.9% 和 60.9%，对照 VLA-JEPA 为 20.8% 和 41.0%。组件消融显示，未来预测、CoD 和强化学习依次把 LIBERO-PRO 平均值从 40.1% 提升到 50.4%、54.4% 和 55.6%。真实双臂实验涵盖颜色匹配、挂杯和分类，宏平均成功率 94.4%；OOD 条件下为 52/60 次成功，每个条件只有五次试验。论文只有一个训练 seed，因此现有置信区间不能反映训练间差异。

#### 实用指南
方法采用 Qwen3-VL-2B、MolmoAct2 动作专家与 DINOv3 特征目标。论文报告 SFT 和强化学习阶段分别使用 128 与 64 张 H100；真实部署测得 ThinkWorld p50/p95 推理时延约 325/333 毫秒、峰值显存 7.07 GiB，测量依赖单张 H100 和指定输入设置。尝试复现时要保持人类腕部与机器人末端轨迹的归一化规则、未来帧时间对齐和具身专属微调一致，并单独报告新机器人上的物理试验。

#### 总结
核心思想：用未来变化校准动作表征
1. 将人类与机器人动作编成共享运动 token。
2. 学习 token 对应的未来视觉状态。
3. 用无动作标签视频的预测一致性强化 CoD。
4. 由专属动作头生成部署控制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.37666v1)
- [arXiv](https://arxiv.org/abs/2610.37666v1)

---

<a id='2610.36466v1'></a>
## [Nemotron-Labs-Diffusion-Uni: Elastic Self-Refining Unified Diffusion LLM with Cross-Paradigm On-Policy Distillation](https://arxiv.org/abs/2610.36466v1)

**Authors:** Shufan Li, Hanrong Ye, Greg Heinrich, Yonggan Fu, Haotian Zhang, Sifei Liu, Aditya Grover, Pavlo Molchanov

**Published:** 2026-10-27

**Categories:** cs.CV

**Abstract:**

We propose Nemotron-Labs-Diffusion-Uni (NLD-Uni), a unified discrete diffusion large language model (DLLM) for visual understanding and generation tasks. Compared with prior unified diffusion models, NLD-Uni introduces two major innovations. First, we propose a unified cross-paradigm on-policy distillation (OPD) framework that successfully transfers the visual understanding ability of autoregressive VLMs and visual generation ability of continuous flow models, to a unified DLLM. Second, NLD-Uni incorporates a novel progressive self-refining diffusion process that gradually corrupts each token across multiple steps during training, as opposed to adopting the conventional masked diffusion formulation that directly corrupts a clean image token to an uninformative mask token. At inference, this design allows NLD-Uni to make intermediate commit to a token, enabling progressive self-refinement for better visual quality. Through extensive experiments, we show that NLD-Uni achieves state-of-the-art performance among unified DLLMs on diverse tasks including visual question-answering, multi-modal reasoning, object grounding, text-to-image generation, and image editing.

### 论文解读
#### 摘要翻译
Nemotron-Labs-Diffusion-Uni提出统一离散扩散语言模型，兼顾图像理解、生成和编辑。它用弹性混合专家视图按任务加载参数，以跨范式在线蒸馏吸收自回归与连续流教师知识，并让掩码图像token反复细化。作者报告多项理解、推理和图像生成能力提升。
#### 方法动机分析
统一离散扩散模型虽能并行生成token，但与专用VLM或连续流图像模型仍有性能差距。直接拿AR或连续流教师做监督会遇到上下文和生成范式不匹配；一次补全并固定token又限制迭代修正。论文分别针对容量使用、教师分布和细化机制设计方法。 关键挑战在于不同范式产生的监督上下文不一致，而掩码扩散又难以像连续生成模型一样修正早期错误。

#### 方法设计详解
Elastic-MoT提供VLM、T2I和交错三种视图，按任务激活理解或生成分支。跨范式OPD在学生自生成rollout上进行：AR教师校正多模态理解，连续流教师修订学生图像，再把修订图像编码成训练目标。图像由离散语义/VQ token表示；自精炼掩码扩散从全掩码状态开始多轮解码，使用token年龄和码本邻域逐步细化已生成内容。速度质量可以通过解码阈值调整。 输入包括文本、图像和任务类型，输出覆盖图像问答、定位、编辑及文本生图。不同视图控制哪些层参与注意力和前馈计算；教师精炼先生成学生分布样本，再把纠正后的答案或图像变成监督，形成“采样—校正—再训练”的流程。

#### 方法对比分析
与统一扩散基线相比，主要差异是引入按任务弹性加载和不同范式教师；与传统离线蒸馏相比，目标来自学生自己的rollout，更贴近实际生成分布。额外收益依赖教师质量、学生生成质量与复杂训练数据管线。
#### 实验分析（精简版）
模型共12B参数，在64张H100上训练26天。所列理解任务平均78.3，较表中离散扩散SOTA高17.1；默认自精炼速度552.58 token/s、MathVista 75.0，放宽阈值后997.95 token/s、73.4。基准集合与测试协议不同，不能把汇总差值理解为统一测试上的直接胜出。 结果同时揭示质量与吞吐的折中；评测跨多个理解与生成基准，且教师、数据和任务构成各异，不宜将所有分数当作同一难度下的统一排名。

#### 实用指南
适合研究统一多模态模型、离散扩散和在线蒸馏的团队。复现需处理多教师推理、学生rollout缓存、图像tokenizer及不同任务视图；资源要求很高。对机器人感知生成有启发，但论文没有机器人闭环证据。
#### 总结
核心思想：理解先行，生成反复细化
1. 按任务选择理解/生成视图。
2. 用不同范式教师监督学生自身样本。
3. 多轮修正离散图像token。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36466v1)
- [arXiv](https://arxiv.org/abs/2610.36466v1)

---

<a id='2610.37626v1'></a>
## [Know, then Draw: A New Learning Paradigm for Visual Generation](https://arxiv.org/abs/2610.37626v1)

**Authors:** Yucheng Mao, Chongjian Ge, Zhuowen Tu, Hao Tan, Hanwen Jiang

**Published:** 2026-10-27

**Categories:** cs.CV

**Abstract:**

Visual generation requires both high-level semantic knowledge and fine-grained pixel details. We propose a learning paradigm that assigns these roles to autoregressive pretraining for knowledge acquisition and diffusion post-training for high-fidelity rendering. Within this framework, we study the objective, representation, and scaling of knowledge acquisition and rendering. Controlled comparisons identify autoregression as the stronger pretraining objective for downstream rendering, improving both learning efficiency and final generation quality. We further find that AR-only generation quality does not necessarily correlate with final performance; semantic quality matters more. This motivates a purely semantic tokenizer trained with language supervision alone, without an image decoder or reconstruction loss. We establish compute-optimal scaling laws for knowledge acquisition and rendering, and find that autoregressive pretraining lowers the fitted asymptotic rendering loss compared with pixel diffusion models trained from scratch. Guided by the laws, we train a 3B-parameter text-to-image model with only 6,600 H100 GPU-hours, achieving 86 on GenEval and 87 on DPG-Bench and validating the proposed paradigm at scale.

### 论文解读
#### 摘要翻译
作者提出“先学语义知识、再学像素渲染”的视觉生成范式：自回归预训练从图像token获取视觉知识，扩散后训练负责生成高保真图像。实验显示语义token比重建导向token更适合作为预训练目标，并据此拟合两阶段算力分配，训练出约3B模型。
#### 方法动机分析
传统视觉自回归tokenizer努力重建像素，因此模型要同时学语义和外观细节；扩散渲染器能够专门处理后者。作者假设两个目标适合不同训练目标：AR擅长获取语义知识，扩散擅长细节渲染，因此不应要求AR token本身能解码成图像。
#### 方法设计详解
预训练时，冻结tokenizer把图像编码为离散视觉token，文本条件AR模型以随机空间顺序预测。Semantic-LM只用冻结视觉语言模型上的语言监督训练码本，没有图像decoder或重建损失。后训练冻结AR骨干，增加像素扩散模块并复用注意力投影，输入噪声图像、文本和图像token，采用flow matching训练速度场。推理时AR先生成视觉token，扩散再据此画图。作者分别拟合AR与渲染阶段的模型/数据/算力缩放关系。 输入是文本提示与训练图像，先由AR产生结构化离散视觉token，再由扩散模型输出像素图像。两阶段各自拟合算力缩放规律，约3:2的预训练/后训练预算来自拟合最优点，而不是把两种不同损失直接相加。

#### 方法对比分析
与从头训练像素扩散或让AR直接生成图像相比，这一方法把语义建模和细节合成分工。Semantic-LM不适合作为独立图像生成器，却可作为扩散条件；两阶段训练也带来训练与推理条件差异。
#### 实验分析（精简版）
Semantic-LM后训练GenEval为64.84，Semantic-Recon为65.51；约3B最终模型用6600 H100 GPU小时，GenEval 0.86、DPG提升至87.78分。跨系统数据和评测不同，且prompt经过改写并基于固定种子选择，作者也未证明算力分配全局最优。 控制实验还发现AR-only生成强弱不能预测扩散后结果；最终比较使用prompt改写与固定种子选择，评测设定对跨系统排名有影响。

#### 实用指南
适合研究视觉tokenizer、语义生成与扩散训练的读者。受控实验约400M参数，最终模型AR阶段150k更新、扩散阶段100k更新，并使用50步Euler采样。复现需准备高算力、冻结视觉编码器和评估prompt协议；本文不提供机器人执行验证。 最终评测还用50步Euler采样，若改采样器或prompt筛选策略，结果不能直接横向对照。 语义token的收益来自受控消融，不应直接推断所有图像任务都可取消重建监督。 模型包含近30亿参数，独立复现成本较高。

#### 总结
核心思想：AR学语义，扩散负责成像
1. 把图像编码成语义token并训练AR。
2. 冻结AR，训练扩散渲染器。
3. 推理先预测token，再逐步生成像素。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.37626v1)
- [arXiv](https://arxiv.org/abs/2610.37626v1)

---

<a id='2610.36646v1'></a>
## [Image Modeling: From Handcrafted Priors to Foundation Image Models](https://arxiv.org/abs/2610.36646v1)

**Authors:** A. Murat Tekalp

**Published:** 2026-10-27

**Categories:** eess.IV, cs.CV

**Abstract:**

This paper unifies six decades of image modeling, evolving from handcrafted mathematical tractable priors to solve inverse problems in imaging to data-driven deep foundation image models for both image generation and solving inverse problems, under a common Bayesian framework that formulates image modeling as an implicit or explicit probabilistic characterization of a data manifold. Every image restoration or generative task fundamentally seeks to answer the same question: What makes an image look like a real image? In the Bayesian framework, this question is encapsulated by the prior probability density p(x), or its more optimization-friendly counterpart, the log-prior log p(x). While early works in solving inverse problems, such as Tikhonov, Total Variation (TV), and Non-local priors relied on explicit handcrafted definitions of log prior to model local smoothness, modern techniques, such as Plug-and-Play (PnP) priors and Denoising Diffusion models, are rooted in algorithmic uncoupling and implicit manifold learning that apply to both inverse problems and conditional image generation. By grounding imaging inverse problems and conditional image generation under a single mathematical engine: the gradient of the log-prior, or the Stein Score Function, we bridge the gap between signal processing and machine learning communities, turning what seems like parallel developments into a single, evolution of image modeling from handcrafted analytical models to models learned from big data, demonstrating that the pursuit of the denoising score is the common thread connecting variational optimization (in MAP estimation) to posterior sampling (in modern foundation image models).

### 论文解读
#### 摘要翻译
本文回顾图像建模从手工数学先验到数据驱动基础图像模型的发展，并用贝叶斯框架连接图像逆问题与条件生成。核心统一量是图像先验的对数概率梯度，即score；它既可指导优化复原，也可用于生成式后验采样。
#### 方法动机分析
去噪、去模糊等逆问题与文本生成看似不同，却都需要表达“什么图像合理”。手工正则化易解释但表达能力有限，深度生成模型更灵活，却可能牺牲观测一致性。作者希望用共同概率语言梳理方法谱系，同时指出两类任务条件信息和评价目标并不相同。 文章试图弥合信号处理和机器学习的术语差异，回应手工正则化难覆盖复杂图像、而生成模型又可能忽视测量一致性的挑战。

#### 方法设计详解
文章从Tikhonov、总变差、稀疏/低秩和非局部先验讲起，解释ISTA/FISTA、HQS和ADMM如何将先验与数据项组合。PnP、RED把去噪器嵌入迭代求解，Tweedie公式建立去噪器与score的联系。随后回顾VAE、GAN、自回归、扩散/SDE、score模型、flow matching、潜扩散和DiT。基础图像模型学习跨任务图像先验，可通过后验采样、潜空间优化、PnP或任务适配器用于复原。隐式神经表示把坐标映射到像素，位置编码或SIREN改善高频细节表达。 文章的分析流程不是训练一个新模型，而是从观测模型和后验出发，追踪先验如何进入目标函数，再比较点估计、迭代优化与随机采样。score作为统一接口解释了去噪器、PnP和扩散模型之间的联系。

#### 方法对比分析
手工先验透明而受限，学习先验表达丰富但会引入数据偏差和幻觉。逆问题有物理前向模型和测量约束；文本生成主要依赖语义条件，两者不能只凭同一个条件概率写法视作等价任务。 相较只按模型结构排列方法，本文用同一贝叶斯问题解释不同路线；创新在教学性整合而非新算法。逆问题适合强调物理约束，条件生成强调语义分布，两者的误差指标也不应混用。

#### 实验分析（精简版）
这是一篇教程综述，没有新增实验、指标或训练设定；证据来自所引用研究、公式推导和概念示例。文中对基础模型可用性和风险的总结不能解读为统一基准上的实证比较。 本文无实验、无新增指标，读者应把它视为概念综述；文中例子说明方法机制，性能结论仍需回到各原始论文核验。

#### 实用指南
适合复原、去噪、生成模型及信号处理读者建立共同术语。实际做逆问题时先明确观测模型与保真要求，再选择优化、后验采样或PnP；同时检查真实细节是否被生成先验改写。文章未提供新的代码或可复现配方。
#### 总结
核心思想：以图像先验连接复原与生成
1. 写出观测似然与图像先验。
2. 用先验梯度做优化或后验采样。
3. 在保真与感知质量间按任务取舍。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36646v1)
- [arXiv](https://arxiv.org/abs/2610.36646v1)

---

<a id='2610.36454v1'></a>
## [SVIn+: Multi-Modal Framework for Robust Underwater State Estimation using 3D Sonar, Visual-Inertial, Water-Pressure, and DVL](https://arxiv.org/abs/2610.36454v1)

**Authors:** Chinmay Burgul, Xiaoming Zhao, Yewei Huang, Michalis Chatzispyrou, Marios Xanthidis, Ioannis Rekleitis, Alberto Quattrini Li

**Published:** 2026-10-27

**Categories:** cs.RO

**Abstract:**

As expansion and scalability in underwater oper- ations becomes more important and challenging, autonomous underwater vehicles (AUVs) become more relevant and the necessity for reliable autonomy more critical. A fundamental challenge in any autonomous underwater operation, from control to safe navigation and mapping, is state estimation; a notorious hard problem to address, especially in low visibility conditions. In this paper, we tackle this problem, by presenting SVIn+, a multi-sensor framework for robust state estimation of AUVs equipped with an IMU, cameras, a water-pressure sensor, a Doppler Velocity Log (DVL), and a compact 3D sonar. This framework extends an established open-source state-of-the-art visual-inertial SLAM, SVIn, by integrating water-pressure and DVL measurements into a tightly-coupled nonlinear optimiza- tion formulation resulting in improved estimates. To leverage the structural information of recently available 3D acoustic sonars while maintaining computational efficiency, we introduce a hierarchical architecture that incorporates estimates from a decoupled front-end via KISS-ICP, treated as relative pose factors within the factor graph. The proposed framework is validated through field deployments in a shipwreck and coral reef, demonstrating drift mitigation and global consistency. To support further research in underwater multi-modal fusion, the code and the datasets can be found on project website https://cmburgul.github.io/projects/svin+/ https://dartmouthrobotics.github.io/svin-plus

### 论文解读
#### 摘要翻译
SVIn+将视觉、惯性、水压深度、DVL速度和3D声呐融合，用于水下机器人状态估计。它通过紧耦合滑窗优化获得高频状态，并把声呐里程计的相对位姿作为约束输入。沉船和珊瑚礁数据上的回访实验显示，多传感器组合能降低漂移。
#### 方法动机分析
水下相机受浑浊、弱光和重复纹理影响；IMU会累积漂移，单一深度或速度传感器也只能约束部分自由度。作者的出发点是让不同传感器互补，同时避免声呐点云把主优化器变得过重。核心假设是声呐前端可独立提供可信的短程相对位姿，加入稀疏因子即可改善全局局部估计。 关键挑战是传感器频率、坐标系和退化模式不同，简单串接容易让一个错误模态拖偏整体状态。

#### 方法设计详解
系统以相机特征重投影和IMU预积分建立视觉惯性基线，再在滑窗中增加DVL速度残差、水压深度残差与声呐相对位姿残差。DVL按外参将传感器系速度转至机体/世界状态；压力传感器给出相对水深，短时间内假定水面恒定。3D声呐以KISS-ICP完成扫描配准，使用上一时刻位姿作运动先验，输出位姿增量而非原始稠密点约束。各残差依据传感噪声协方差加权优化。 输入为异步传感器观测，输出为滑窗中的位姿、速度和偏置估计；声呐模块先配准并输出相对变换，局部优化器再联合视觉、惯性、深度和DVL残差。每项误差按传感噪声加权，因而异常模态的影响取决于噪声设定。

#### 方法对比分析
SVIn+的差异在于把水下特有的深度、DVL和声呐信息纳入统一估计，而非仅用视觉惯性或单独后处理声呐轨迹。模块化降低集成负担，但声呐因子质量依赖前端匹配，和全球回环识别互补而不能取代。 相比视觉惯性基线，它增加水下特有的垂向和速度约束；相比直接使用声呐点云，它先做前端配准再传递稀疏位姿因子，适合关注实时性的移动机器人。

#### 实验分析（精简版）
在BajanQueen-01的ArUco回访重投影误差上，SVIn+为0.21米，SVIn2为0.53米；仅深度为0.29米，仅DVL为0.52米。数据没有全球轨迹真值，回访误差只能近似表示累计漂移，潮汐变化也被短时恒水面假设忽略。 作者还在第二处珊瑚礁序列展示地图和长距离轨迹，但缺少独立全局真值，因此不能据图形平滑度推断绝对精度。

#### 实用指南
适用于水下定位研究者搭建多模态因子图。需仔细标定相机、IMU、DVL、深度计和声呐外参，并复核各频率的时间同步与噪声模型。作者报告在i7-10510U笔记本上实时运行；部署环境若算力或水体条件不同，应重新测前端耗时和传感退化。
#### 总结
核心思想：稀疏融合水下多源约束
1. 传感器前端生成视觉、惯性和声呐观测。
2. 将深度、DVL与声呐转为滑窗残差。
3. 联合优化位姿并用回访评估漂移。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36454v1)
- [arXiv](https://arxiv.org/abs/2610.36454v1)

---

