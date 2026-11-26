time: 20261126

# Arxiv Computer Vision Papers - 2026-11-26

## Executive Summary

## 2026-11-26 ArXiv 计算机视觉论文执行摘要

本期纳入 10 篇论文，主题集中在视觉与机器人学习的交叉地带。一个共同趋势是把视觉表征放进可检验的物理或部署情境中：研究者不只追求图像或视频预测质量，也评估几何一致性、接触行为、规划闭环表现，以及策略在扰动和不同机器人形态上的适用性。另一条主线是以未来状态或潜在动作组织视频信息，降低机器人策略对昂贵在线模型的依赖。

**值得重点关注的工作**包括 MoGe-4D，它将动态几何重建、相机运动和稠密跟踪置于共享查询框架，并在六个短序列跟踪基准上报告了竞争性结果；其长序列退化、细结构误差和全局注意力成本仍是限制。R2R 对已发布自动驾驶模型进行规划与感知的针对性迁移修正，仿真规划成功率和离线车辆检测指标均有提升，但实车感知结果并非闭环安全验证。ViTacLab 把接触锚定的触觉仿真与跨具身基准结合，降低了力误差；真机任务的提升幅度有限，盲抓表现也提示仍有差距。RESET 则以单次示范重建可交互场景和冻结的成功判据，展示跨机器人练习的潜力，同时报告了较高的大模型调用成本和各平台表现差异。

**新兴方向**主要出现在机器人世界动作建模与训练数据设计。LeanWAM、PreWAM 和 Agent-State LAM 分别研究如何从冻结视频模型蒸馏动作策略、设计潜在未来表征，以及让紧凑状态同时服务潜动作推断和未来预测；结果显示表征内容与时间尺度会影响性能，但目前实体评估规模有限，跨机器人泛化仍待验证。SimPWM 通过显式模拟物体抓放来约束生成视频，冻结物体实验揭示出外观逼真与物理状态遵循之间的差异。对抗遥操作则探索由人主动制造可恢复困难来补充训练数据，初步结果支持其用于恢复行为学习，但采集和标注的人时成本尚未核算。

**阅读建议：**关注动态三维感知的读者可优先读 MoGe-4D；研究机器人策略表征与视频预训练的读者可将 LeanWAM、PreWAM、Agent-State LAM 对照阅读；关注部署可靠性和评估方法的读者可读 R2R 与 ContinualDrive，后者显示离线误差改善不必然带来闭环收益。若重点是具身交互与仿真迁移，可读 RESET、ViTacLab 和 SimPWM。相关结果多来自仿真、离线视频或规模有限的真机试验，解读时应结合各论文注明的评估边界。

---

## Table of Contents

1. [Beyond the Benchmark: Localizing Released Autonomous Driving Models to Real Vehicles](#2611.31814v1)
2. [RESET: One Demonstration, One Verified World, Self-Improved Skills Across Embodiments](#2611.31894v1)
3. [ViTacLab: Physically Consistent Visuo-Tactile Simulation, Benchmarking, and Representation Learning for Robotic Manipulation](#2611.32476v1)
4. [ContinualDrive: A Deployment-Anchored, Multi-Axis Benchmark for Continual Updates of Learned Driving Planners](#2611.32529v1)
5. [Simulator-as-Proxy World Model for Robot Manipulation](#2611.32786v1)
6. [Exploratory Midtraining for Robot Policies](#2611.33130v1)
7. [LeanWAM: Learning Latent World Action Models from Frozen Video Models](#2611.31740v1)
8. [PreWAM: Exploring the Design of Latent Future Prediction for World-Action Modeling](#2611.31901v1)
9. [Learning Agent States for Latent Action Modeling](#2611.32017v1)
10. [MoGe-4D: Efficient, Flexible, and Unified Dense 4D Reconstruction and Tracking](#2611.32198v1)

---

## Papers

<a id='2611.31814v1'></a>
## [Beyond the Benchmark: Localizing Released Autonomous Driving Models to Real Vehicles](https://arxiv.org/abs/2611.31814v1)

**Authors:** Morui Zhu, Deyuan Qu, Qi Chen, Qing Yang

**Published:** 2026-11-24

**Categories:** cs.RO, cs.CV

**Abstract:**

Released autonomous-driving models achieve strong benchmark performance, yet deploying them on a new real vehicle remains challenging. We study this deployment gap and find that released models can remain coupled to assumptions from their source sensing, data, and execution processes even after conventional adaptation. We introduce Released-to-Real (R2R), a deployment-localization methodology that identifies such source-system coupling and corrects only the affected components while preserving the released model. For data and behavior adaptation, we find that the mismatch between trajectory-level optimization and receding-horizon execution enters through the ego state rather than the observation, and introduce Human-Anchored Reinforcement Learning (HA-RL), which corrects it on recorded target driving without the costly rollouts of generative closed-loop reinforcement learning (RL). For physical sensor adaptation, R2R localizes and corrects source-dependent spatial and temporal representations with frozen perception weights, requiring neither target annotations nor retraining. HA-RL improves the closed-loop success rate of a released diffusion-based planner from 82.2% to 95.2%, while sensor adaptation across multiple released detectors and three real sensor rigs improves the vehicle mean average precision (mAP) of a released multi-camera detector from 0.041 to 0.420 on a five-camera target rig. These results provide a practical path for preparing released models before deployment on new real vehicles. Our project page is publicly available at https://released-to-real.github.io.

### 论文解读
#### 摘要翻译
已发布的自动驾驶模型在基准上表现良好，但换到新车辆仍可能失败，因为模型会保留原传感器、数据和执行流程的假设。作者提出 Released-to-Real（R2R）：先定位这种源系统耦合，再只修正受影响部分。规划侧提出人类锚定强化学习（HA-RL），在目标车记录数据上处理轨迹级优化与滚动执行的差异；感知侧在冻结权重下校正空间与时间表示，无需目标标注或重训。扩散规划器闭环成功率由82.2%升至95.2%；五摄像头目标架上的车辆检测mAP由0.041升至0.420。

#### 方法动机分析
开源模型迁移时，正确标定和常规微调仍未必消除网络内部的旧假设。规划器按整条轨迹得分，但车辆只执行短片段并反复重规划，两种过程可能使开环分数提升而安全性下降。作者的核心假设是，规划失配可通过自车状态单独定位；相机迁移的误差则可由固定模型、改变几何或更新时间间隔的探针暴露。

#### 方法设计详解
R2R先做传感器接口对齐（SIA），将目标相机重渲染为模型训练时的相机。规划流程先用目标车记录做模仿学习，并只用历史位姿构造自车状态、扰动当前加速度，减少模型从驾驶员动作中“偷看”答案。HA-RL在固定的真实记录帧上执行规划器，把实际执行首段后的自车状态反馈给下一次规划；它不生成新视频。奖励同时看滚动执行质量及首段速度变化是否落在人类驾驶范围内，再用组相对策略优化更新低秩适配器。感知流程冻结检测器权重，通过内容无关输入找出位置先验；按相机高度重排位置特征，或修正稀疏锚点的高度几何（SRA）。随后将时序记忆按源相机的帧间隔更新（TRA），避免目标帧率改变记忆含义。修正位置随检测架构而异，依赖源和目标相机标定。

#### 方法对比分析
常规模仿学习依据人类轨迹，无法保证优化后策略在反复执行下安全；轨迹级强化学习仍按整条预测轨迹打分。生成式闭环强化学习能模拟状态和图像共同变化，但每次rollout成本高。HA-RL只让自车状态沿策略执行而图像保持记录帧，并加入人类速度锚定。相比目标域重训或有监督跨相机适配，SRA/TRA不改模型权重，但需识别架构中具体的几何与时序编码。

#### 实验分析（精简版）
在90个、每个4秒的生成式闭环事件中，HA-RL将释放扩散规划器成功率从82.2%提高到95.2%，碰撞率从4.4%降至1.9%；同一设置下轨迹奖励RL成功率仅78.1%。HA rollout约0.9秒，生成式rollout约17秒。感知检测在真实传感器记录上离线评估：冻结的时序位置嵌入模型在五摄像头目标架上的车辆mAP由0.041升至0.420。闭环规划结果来自世界模型仿真，感知结果使用LiDAR伪真值，不等同于实车道路安全验证。

#### 实用指南
项目页公开；论文使用公开检测器权重并报告了可复现的规划设定：IL训练1000步、学习率2×10⁻⁵、batch为8；HA-RL训练低秩适配器（rank 16），4张GPU更新100步。迁移时需准备目标驾驶记录、源/目标相机标定及可修改的表示或记忆接口；先做接口重渲染，再用单变量探针判断是否存在空间或时序残留。真实车辆栈结果是离线回放，完整实车闭环仍需另行验证。

#### 总结
核心思想：先定位旧假设，再局部校正。
1. 重渲染目标输入，对齐已发布模型的传感器接口。
2. 用固定图像/权重的受控探针定位规划、空间和时序失配。
3. 在目标记录上修正自车状态策略或表示/记忆，再独立验证闭环行为。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.31814v1)
- [arXiv](https://arxiv.org/abs/2611.31814v1)

---

<a id='2611.31894v1'></a>
## [RESET: One Demonstration, One Verified World, Self-Improved Skills Across Embodiments](https://arxiv.org/abs/2611.31894v1)

**Authors:** Zheng Sun, Zhuo Li, Zhihao Li, Hengyi Sim, Ge Wang, Ziyue Zhu, Mingkai Jia, Fei Chen

**Published:** 2026-11-24

**Categories:** cs.RO

**Abstract:**

Large language model (LLM) coding agents can iteratively self-improve robot manipulation skills in simulation benchmarks, yet real-world tasks provide neither ground-truth success scorers nor resettable practice environments. Our key observation is that a single real demonstration simultaneously specifies scene geometry, defines the terminal success criterion, and serves as a physical reference trajectory for verifying world fidelity---all decoupled from the source robot's kinematics. We present {RESET}: we reconstruct a contact-rich world and derive a frozen read-only criterion from one demonstration ({R}ebuild); check that world against the demonstration by open-loop replay and silhouette consistency ({E}xamine); evolve modular white-box task graphs and skill code from per-node failure evidence ({S}elf-improve); deploy frozen code on real robots with zero runtime LLM calls ({E}xecute); and share the verified world and criterion with unseen robots without new demonstrations ({T}ransfer). On 8 manipulation tasks, each rebuilt from one real demonstration, RESET achieves 97% average success on unseen layouts in simulation, whereas zero-shot code reaches 27%, per-task fine-tuned pi0.5 policies 31%, and Diffusion Policy 50%; injecting reconstruction defects shows that practice tolerates many reconstruction errors, that geometric errors can still stall or mislead it, and that the check flags most of them while physical randomization covers the rest. Without any target demonstrations, 3 heterogeneous robots acquire the same task skills in the shared world: a robot with similar arms copies 95% of its skill functions unchanged from the source robot, while robots with different arms and hands write their own. In real-robot experiments, the resulting programs run on 4 robots and succeed in 80% of 280 trials.

### 论文解读
#### 摘要翻译
RESET利用一次真实示范重建并检验可交互世界，从中提取成功条件，让编码智能体反复练习生成机器人技能，再将同一任务环境交给其他形态机器人。最终代码部署时无需调用大模型。
#### 方法动机分析
机器人代码自改进依赖可重置环境和可靠评分，而真实任务通常没有这些条件。作者假设一次成功示范既提供场景几何与目标状态，也能提供检验重建质量的物理参照，从而把任务要求与机器人运动学分开。
#### 方法设计详解
输入是一次RGB-D、关节和夹爪记录及机器人URDF。系统先分割首帧物体、配准深度并建立含空腔和关节的物理场景；根据末态生成几何关系和释放静止条件，经人确认后冻结评分器。随后以记录的关节命令开环重放，比较接触产生的物体运动和抓取，并以渲染轮廓与图像掩码的IoU检查视觉几何。质量、摩擦难以由一次示范识别，练习时随机化。
控制器由任务图、技能库及其绑定组成。编码智能体读取节点状态、失败原因和关键帧，修改任务步骤或技能几何；仅当独立留出布局表现改善才采纳。诊断和留出各用20个布局，最后用50个未见布局测试。推理阶段运行冻结代码，每个节点前由分割和深度估计物体位姿，不调用大模型。目标机器人提供自己的模型、相机标定及末端接触偏移，在共享世界中继续练习，输出冻结程序。
#### 方法对比分析
相比直接模仿轨迹，该方法共享环境和任务判据，使异构机器人能编写各自技能；相较只看最终成败的搜索，节点反馈更利于定位修复。但世界检查只能筛除部分错误，不能保证物理真实性。
#### 实验分析（精简版）
源机器人八任务、每任务50个未见仿真布局上，平均成功率97%，零样本代码为27%，使用每任务100条成功轨迹训练的Diffusion Policy为50%。四平台真机共成功223/280次，但人形G1仅13/40，显示全身控制造成的仿真差距仍大。对照实验中，微调π0.5也使用每任务100条轨迹，训练6000步；这一数据规模有限，不能据此推断所有策略训练设定下都存在同样差距。几何缺陷检查检出了16个破坏技能案例中的13个，并非万无一失。
#### 实用指南
论文给出项目页面，但未明确说明代码开源状态。复现需准备深度相机、机器人模型、物体重建与标定，并单独确认终态判据；每任务每机器人练习约100–320次仿真、消耗约3000万大模型token。迁移到新平台仍需硬件适配和验证。
#### 总结
核心思想：一次示范构造共享练习场。
1. 采集示范，重建场景与成功标准。
2. 校验世界，从节点失败修订代码。
3. 新机器人在共享世界练习，部署冻结程序。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.31894v1)
- [arXiv](https://arxiv.org/abs/2611.31894v1)

---

<a id='2611.32476v1'></a>
## [ViTacLab: Physically Consistent Visuo-Tactile Simulation, Benchmarking, and Representation Learning for Robotic Manipulation](https://arxiv.org/abs/2611.32476v1)

**Authors:** Linjing You, Zeming Yang, Zhenhao Shen, Shengqiang Xu, Guanda Li, Wenxuan Li, Jiyu Guo, Jasper Lu, Kyle Xu, Shawn Xie, Junjie Zhu, Siyu Wu, Jian Liu, Changhao Zhang, Rongchong Li, Zeyi Li, Hengshuang Zhao, Ruihai Wu

**Published:** 2026-11-24

**Categories:** cs.RO

**Abstract:**

Tactile sensing is essential for contact-rich robotic manipulation, but existing visuo-tactile simulators trade physical fidelity for scalability, and benchmarks largely focus on two-finger grippers. We present ViTacLab, a framework comprising ViTacSim, ViTacBench, and ViTacPolicy. ViTacSim uses a tri-modal tactile primitive: a global sparse contact-force lattice, a local dense force array, and tactile images. Anchored to PhysX contact-solver outputs, it corrects normal inconsistency through local stiffness alignment and reconstructs tangential forces under decoupled slip-stick regimes while preserving GPU parallelism. A joint projection preserves the sparse wrench and enforces local friction cones. ViTacBench spans parallel grippers, dexterous hands, and bimanual systems, with tasks ranging from vision-dominant to tactile-dominant regimes. ViTacPolicy aggregates variable numbers of heterogeneous tactile inputs using spatial-pose embeddings and a Transformer, and aligns force-array and tactile-image modalities through a two-stream VQ-VAE with a shared codebook. Controlled Hertzian-reference indentation and real-sensor sliding tests reduce normal and tangential force RMSE from 2.31 and 1.85 N to 0.42 and 0.53 N, respectively. Throughput reaches 2,482.9 tactile-pipeline FPS at 256 parallel environments. Simulation and real-world experiments show gains on several contact-rich tasks, while low absolute success on tactile-dominant tasks highlights the remaining difficulty of purely tactile control.

### 论文解读
#### 摘要翻译
ViTacLab 将触觉仿真、分层基准和异构传感策略结合，面向接触丰富的机器人操作。它以物理求解器接触力为锚点生成多种触觉信号，并通过共享表示连接力阵列与触觉图像。实验显示力预测和部分操作任务改善，但触觉主导任务仍很困难。
#### 方法动机分析
遮挡与精细接触使视觉不足；仅用穿透深度的仿真会在高负载饱和，仅用速度估算静摩擦也欠定。作者假设求解器提供的稀疏接触可约束密集触觉，传感器位姿可帮助策略处理变化的布局。现有评测又常局限于单一夹爪，难以区分视觉受损时触觉到底提供了多少信息，因此需要跨平台和分级任务检验。
#### 方法设计详解
首先从 PhysX 提取接触位置、力、法向、深度和切向速度，投影到传感器获得稀疏接触。局部力深比估计刚度，校正法向形变；切向按滑移与黏着分区，分别用锚点校准及高斯插值。最终联合投影使密集力的合力和力矩匹配求解器，并满足局部库仑摩擦锥，再生成力阵列与光学图像。
策略分别编码触觉输入，将每个传感器位置和朝向编码加入特征，Transformer 聚合后与视觉及本体状态交给扩散策略。预训练时，配对力阵列和图像经各自编码器与共享离散码本重建，并对齐全局特征。每项任务采用100条策略训练轨迹和另100条预训练轨迹，仿真评估使用五个随机种子。在 RTX 4090D GPU 上，256个并行环境的触觉管线吞吐为2482.9 FPS。
#### 方法对比分析
相较只从穿透或速度生成触觉，方法显式约束法向、静摩擦和整体力矩；相比固定布局融合，位姿编码支持不同数量和位置的传感器。基准还覆盖夹爪、灵巧手和双手，并按视觉依赖程度分级。
#### 实验分析（精简版）
Hertz 压入参考下，法向力 RMSE 从穿透基线的2.31降至0.42 N；真实传感器滑动测试中切向 RMSE 从1.85降至0.53 N。真实夹爪三任务每项50次，完整方案平均成功率61.3%，无预训练为48.7%；单任务差异尚不显著。盲抓成功率仅3%。仿真比较使用五个随机种子，完整预训练并非每项任务最优；力和力矩一致是相对求解器的构造约束，并不等同于真实世界的完全准确。
#### 实用指南
论文未明确提供代码发布地址。复现需配置 PhysX 接触报告、传感器几何及摩擦参数，采集配对力阵列和图像，并对真实光学传感器随机化光照与标记噪声。密集力场和图像保真尚未直接验证；迁移新物体或硬件时应重新标定并实测。
#### 总结
核心思想：物理锚定触觉并统一多模态。
1. 提取求解器接触并校正力场。
2. 生成力阵列与图像，通过共享码本对齐。
3. 融合传感器位姿和视觉，输出操作动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.32476v1)
- [arXiv](https://arxiv.org/abs/2611.32476v1)

---

<a id='2611.32529v1'></a>
## [ContinualDrive: A Deployment-Anchored, Multi-Axis Benchmark for Continual Updates of Learned Driving Planners](https://arxiv.org/abs/2611.32529v1)

**Authors:** Zirui Li, Zhiqi Mao, Yangang Ren, Heye Huang, Haohan Yang, Qihang Huang, Zhan Chen, Chen Lv

**Published:** 2026-11-24

**Categories:** cs.RO, cs.AI

**Abstract:**

Learned driving planners keep being updated after deployment as new logs arrive, yet continual learning evaluations rarely ask whether an update improves on the deployed model, and their verdicts rest on unstated evaluation choices. We present ContinualDrive, a deployment-anchored benchmark for post-deployment continual learning of driving planners. It updates a deployed planner on real-world nuPlan logs one scenario-type group at a time, evaluates the same checkpoints in open and closed loop, makes four measurement axes explicit (evaluation loop, reference checkpoint, temporal aggregation and stream protocol), and anchors a three-valued release verdict to the deployed checkpoint. We study two planners (PlanTF and Diffusion Planner) under full fine-tuning and adapter or LoRA updates, with and without replay or gradient projection, in three task orders. Verdicts can reverse or vanish when one axis changes: open-loop ADE ranks the final PlanTF model above the deployed one in half of its core-suite runs, whereas every closed-loop score ranks it below; backward transfer misses this loss, which on average is already present when each group has just been learned; and regime rankings vary far less across training seeds than across task orders. In the core suite of full and adapter updates, no regime measurably improved on the deployed planner in closed loop; on a panel more than ten times larger every evaluated final checkpoint trails it, and joint training on all groups recovers much of the loss without exceeding its nuPlan closed-loop score. We distill these findings into a reporting contract and will release the code and all checkpoints.

### 论文解读

#### 摘要翻译
ContinualDrive 是面向已部署驾驶规划器持续更新的基准。它把 nuPlan 新日志按场景组依次用于离线更新，并在同一批检查点上评估开环与闭环表现。论文明确评估循环、参照检查点、时间聚合和任务顺序四个维度，以原部署模型为基准给出更好、无可检测变化或更差的发布判断。PlanTF 的开环 ADE 在一半核心运行中优于原模型，但所有闭环分数都更差；核心套件没有任何方案在闭环上可测地超过原模型。

#### 方法动机分析
车辆投入使用后还会积累新日志，模型更新可能学到新场景，也可能让原有驾驶能力退化。只报告最终模型相对学习末端的遗忘量，会漏掉更新过程早已发生的损失；离线轨迹误差也不能替代真实闭环表现。作者的核心假设是：发布判断应回答“是否胜过当前部署模型、是否让原本通过的场景变成失败”，并明确说明评估如何循环、以谁为参照、怎样汇总、按什么顺序训练。

#### 方法设计详解
输入是两个已训练规划器、按七类场景分组的 nuPlan 新日志，以及一组固定开发样本。每次只用当前组更新，再保留该阶段检查点。对照方案包括全量微调、冻结骨干的瓶颈适配器、适配器加 512 条/组的随机回放，以及适配器加梯度投影。每组训练 20 个 epoch，使用 AdamW，峰值学习率 1e-4，64 个样本的微批次累积 4 次；全数据顺序更新每个运行共 32,420 次优化更新。

评估端对同一检查点同时做开环轨迹误差评估和 nuPlan 20 秒闭环仿真。闭环中其他交通参与者既按日志重放，也可由 IDM 对车辆作反应；报告闭环分数 CLS、通过碰撞与道路约束并达到最低进度的成功率 SRv0。核心闭环面板每个场景组有 8 个目标，共 64 个场景。基准从原部署点 C0、学习终点、最佳或前一检查点等不同参照汇报结果，并比较最终、阶段平均和最差阶段。配对场景 bootstrap 给出区间，另统计 C0 通过而更新后失败的目标。

#### 方法对比分析
与主要看 BWT 等遗忘量的持续学习评测相比，ContinualDrive 把部署模型 C0 作为不可忽略的参照，并关注逐场景闭环回归。它不提出新训练算法或排行榜，而是组合开环与闭环、多个参照和汇总方式、不同数据顺序及多种更新策略，形成发布报告规范。适用于研究驾驶模型更新的团队；用于真实车队前仍需独立日志、更多训练种子和实车验证。

#### 实验分析（精简版）
24 个核心运行中，没有方案的闭环结果可测地优于 C0。PlanTF 的 12 次运行里，6 次开环 ADE 点估计更好，但闭环点估计 12 次全部下降；例如 Replay 的 ADE 降低 1.526 米，CLS-NR 却从 0.528 降到 0.454。扩展面板含 630 个目标场景，所有最终检查点仍低于 C0，但该面板与核心面板来自同一开发集。主要优势是发现单一指标可能给出相反结论；局限是场景面板较小、规划器只有两种、无实车实验。

#### 实用指南
论文称将发布代码与检查点。复现时需要精确固定 C0、日志分组、更新预算、采样和 EMA 设置，并至少比较正序、逆序及非镜像顺序；同时保存每一阶段检查点，报告闭环回归及最终、阶段平均和最坏结果。对其他驾驶栈可迁移评估契约，但具体数值不应直接外推到实车、端到端感知系统或新日志分布。

#### 总结
核心思想：更新必须胜过部署基线

1. 固定当前部署模型和新增日志。
2. 按场景组逐步更新并保留检查点。
3. 同时测开环、闭环和场景级回归。
4. 跨任务顺序汇报最终及阶段表现。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.32529v1)
- [arXiv](https://arxiv.org/abs/2611.32529v1)

---

<a id='2611.32786v1'></a>
## [Simulator-as-Proxy World Model for Robot Manipulation](https://arxiv.org/abs/2611.32786v1)

**Authors:** Tingting Liao, Mingjia Huo, Hao Li, Zhengzhong Liu, Eric P. Xing, Guangyi Liu

**Published:** 2026-11-24

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

Video world models for robots are increasingly conditioned on a simulated or rendered proxy of the scene, on the assumption that the video will show what the proxy shows. Testing this assumption requires a proxy whose state is explicit and can be perturbed. We present SimPWM (Simulator-as-Proxy World Model): given a calibrated camera, it represents, from the first frame alone, the robot by its 3D model and the objects the gripper may touch as 3D points; a rule-based simulator with no trained parameters moves them under the actions, and a video model renders the result. SimPWM returns the realistic video together with the simulated state. We then freeze the simulated objects while the arm moves as recorded and check whether the video still shows the object lifted. On DROID episodes from unseen setups, the answer differs for the arm and for the objects. With each of three base video models, the arm follows the proxy (robot IoU 0.75-0.78, against at most 0.69 for end-to-end models); rendering the robot alone already gives this. What the simulated objects add is the object state: against a robot-only proxy, they cut missed pick-ups from 54% to 38%, and the video follows them only with some base models: on rigid objects, freezing them lowers the lift rate of MiniMax-H3, fine-tuned on episodes whose simulation agrees with the real video, from 42% to 10%, but that of LTX-2.3, trained on the same episodes, only from 57% to 49%, as if it inferred the grasp from the arm motion; more than half of the remaining lifts of MiniMax-H3 are of cloth and rope, which the simulator carries rigidly. With any base model, the simulated state records what the simulation did; under perturbed actions, a video that disagrees with it reveals that the video ignores the simulated state.

### 论文解读
#### 摘要翻译
SimPWM 将机械臂及可能接触的物体表示为可操作的三维场景，用无训练参数的规则模拟动作，再由视频模型生成逼真画面，同时输出模拟状态。冻结物体实验发现，机械臂通常遵循代理，但物体是否遵循取决于视频模型。
#### 方法动机分析
端到端视频可能看似合理，却让手臂偏离指令或让物体凭空被抓起。纯仿真又难以重建真实外观。作者假设粗几何与抓放规则可决定主要运动，视频模型负责外观与细节；显式状态还能进行反事实检验。物体原有纹理来自首帧，未被观察到的区域则需要生成模型推断，因此代理的准确性和模型对代理的依赖要分别衡量。
#### 方法设计详解
输入为标定相机首帧、关节轨迹和夹爪指令。系统用机器人网格与正向运动学还原机械臂，通过分割和深度将物体转为点集；闭合点投影到目标时，沿视线缩放点集以对齐夹爪深度。规则根据夹爪内点数决定抓取，之后将点集固定于末端坐标系，释放时落到场景高度图。逐帧渲染机器人及移动物体，得到RGB代理；视频模型接收代理、首帧和任务文本，补全纹理、阴影和遮挡区域。模拟状态与生成视频分别输出。抓取时系统选择夹爪范围内点数最多的候选物体，并在夹爪张开时解除绑定；模拟不需要学习参数，但阈值需要开发集设定。主模型在845条模拟与真实一致的DROID片段上微调。
#### 方法对比分析
相比仅渲染机器人，物体点集提供可修改的抓取状态；相比端到端视频，模拟结果可直接检查。与精细物理重建相比，点集避免逐物体建模，但规则只覆盖有限的抓放行为。
#### 实验分析（精简版）
在293条来自未见配置的DROID片段中，相比仅机器人代理，完整RGB代理将真实拾起任务的漏拾率从54.1%降至37.6%。在96个刚体目标上冻结物体后，H3-C的提起率由42%降至10%，LTX版本由57%降至49%，显示不同基础模型对物体条件的依赖差异。指标涉及视觉模型评判，动作只是回放。
#### 实用指南
代码地址为 https://github.com/TingtingLiao/SimPWM 。复现需机器人网格、准确相机标定、深度与物体分割，注意闭合点误差会导致抓错。相机标定质量尤其重要：论文用多帧机器人掩码对每个片段拟合，且部分帧来自评估窗口。其他机器人需配置运动学；把冻结条件用于测试视频模型的控制依赖，也可迁移到其他场景。
#### 总结
核心思想：用可干预仿真约束视频。
1. 从首帧重建机器人与物体点集。
2. 用动作驱动规则抓放并渲染代理。
3. 生成视频，再扰动物体状态检验一致性。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.32786v1)
- [arXiv](https://arxiv.org/abs/2611.32786v1)

---

<a id='2611.33130v1'></a>
## [Exploratory Midtraining for Robot Policies](https://arxiv.org/abs/2611.33130v1)

**Authors:**  Kshitiz, Zheyuan Hu, Robyn Wu, Yizhou Li, Yunchu Zhang, Vattanary Tevy, Kavya Tayal, Shawn Yin, Aykut Onol, Kushal Arora, Dhruv Shah, Abhishek Gupta, Zackory Erickson, Aviral Kumar

**Published:** 2026-11-24

**Categories:** cs.RO, cs.LG

**Abstract:**

Robot policies predominantly learn from human-collected data emphasizing competent behavior, limiting coverage of the low-level behaviors needed to support autonomous improvement. We study how a form of midtraining can be used to build coverage over diverse low-level behaviors that connect challenging states of the robot and its environment to states from which further progress is possible ("connected behavioral coverage"). We also examine how this coverage supports downstream adaptation and online improvement. To this end, we introduce $β$-Data, a dataset collected via adversarial teleoperation, where a human adversary reversibly perturbs the environment while a human operator continues toward completion without resetting. We relabel trajectory segments with local outcomes in natural language and finetune a pretrained policy conditioned on recent action history. The resulting policy exhibits sustained use of more low-level behaviors. Across manipulation tasks in simulation and on three real-robot platforms, our approach yields more favorable data scaling laws, better zero-shot generalization, and more effective use of additional rollout time than training on standard demonstrations. During online finetuning, these policies need fewer rollouts to collect successful experience and reach higher success rates, suggesting that midtraining can prepare robots for both more robust task execution and continued autonomous improvement. Our project page and robot videos are available at https://kshitiz-1225.github.io/exploratory-midtraining-robot-policies/.

### 论文解读

#### 摘要翻译
机器人示范通常强调熟练完成，缺少应对困难状态的低层行为。本文通过对抗遥操作采集β-Data，并用局部结果语言和动作历史训练策略，使其更能恢复、泛化并从自身经验改善。

#### 方法动机分析
只增加任务或初始状态，并不保证机器人掌握从失败状态继续推进的动作。作者假设，展示扰动、恢复到重新推进的完整连接过程，可以扩充可用行为。这里关注的是给定目标后如何抓取、调整和重试，而不仅是让规划器选择下一步。噪声扰动虽能扩大动作分布，却不一定有针对性地形成连贯的恢复示范。

#### 方法设计详解
采集时操作员持续完成任务，对抗者可逆地移动物体或改变目标，操作员从新状态继续，不重置。将轨迹切成局部结果片段并标注语言；图像、机器人状态、局部指令和最近120步动作进入预训练π0.5，输出未来60步动作块。历史相对当前状态编码成四个token，提供此前尝试的上下文。中训练对图像编码器和动作专家全量微调，语言骨干使用rank-16 LoRA，训练10轮。覆盖度指标将片段首尾映射到抽象状态，统计不同有向转移；相同边的不同动作不会重复计数。分层执行时，高层规划器提供当前子任务。

#### 方法对比分析
标准示范主要展示熟练路径；对抗采集主动制造可恢复困难，保留连续恢复过程。与依赖现有策略失败的人工纠正相比，它可以独立于待适应策略采集。局部语言指出目标，动作历史帮助选择如何继续。

#### 实验分析（精简版）
仿真20个装箱任务每项50次，β-Data总体成功率69.7%，标准示范为54.7%。实体双臂装箱零样本成功12/20，对照为4/20；在线成功轨迹微调后升至17/20。最小数据预算下反而不如标准示范，最难重排仍很困难；实体试验规模有限。更长的执行时间也有帮助：在5.2小时训练预算下，β-Data从两分钟的55.5%升至六分钟的69.3%，对照从47.8%升至58.7%。这些结果表明策略能利用额外时间继续尝试，但不能证明任意任务或机体都会受益。

#### 实用指南
基础模型π0.5已有公开版本，论文提供项目页面和视频；本文数据与训练代码是否发布需核实。复现应匹配录制时间、分段标注、动作历史和任务指令，并记录双人采集及标注成本。训练中对历史动作加少量噪声，并以0.2概率整体丢弃历史；局部指令需要与测试时的规划器输出保持一致。思路可用于机械臂故障恢复；其他领域需验证扰动是否可逆和行为是否安全。

#### 总结
核心思想：用可恢复扰动扩充行为。

1. 对抗者制造困难状态。
2. 操作员连续恢复并完成任务。
3. 标注局部结果，结合动作历史中训练。
4. 执行并筛选成功经验进行在线微调。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.33130v1)
- [arXiv](https://arxiv.org/abs/2611.33130v1)

---

<a id='2611.31740v1'></a>
## [LeanWAM: Learning Latent World Action Models from Frozen Video Models](https://arxiv.org/abs/2611.31740v1)

**Authors:** Shaolong Li, Hongze Fu, Joyce Chai, Yinpei Dai

**Published:** 2026-11-24

**Categories:** cs.CV, cs.RO

**Abstract:**

World action models (WAMs) typically adapt large pretrained video generators to jointly predict future observations and robot actions, incurring substantial training and deployment costs. However, it remains unclear which future representations a policy actually needs and how compactly they can be represented. Our probing study shows that complementary future representations can be compressed into a few tokens while retaining most of their action-decoding performance. Motivated by this finding, we introduce LeanWAM, a latent WAM that transfers video priors from a frozen video generator to a lightweight policy via two-stage training. In the first stage, a Future Extractor compresses observed transitions into 32 predictive tokens. These tokens condition the frozen generator and are jointly shaped by action, RGB, semantic, and geometric supervision, while a compact readout of the generator's hidden states supplies complementary world tokens. The second stage distills both representations into the policy, which learns to predict them from current observations while generating actions. With 1.4B parameters and no large-scale embodied pretraining, LeanWAM outperforms strong vision-language-action (VLA) and WAM baselines on LIBERO, LIBERO-Plus, and RoboTwin~2.0. It predicts an action chunk in 25.1\,ms on an A100 GPU, 12.3--41.5$\times$ faster than representative WAMs. These results demonstrate that compact predictive supervision can transfer video priors to robust and efficient control.

### 论文解读
#### 摘要翻译
LeanWAM 将冻结视频生成器的先验迁移到轻量机器人策略。它先从未来轨迹压缩出预测码并提取世界特征，再将两种表示蒸馏给仅使用当前观测的策略，在多个操作仿真基准上取得较高成功率和较低推理延迟。
#### 方法动机分析
世界动作模型常在训练和部署时运行大型视频模型，但视频包含大量对控制无关的背景与纹理。作者假设动作所需的未来信息可以紧凑表示：探针显示，融合多种未来特征后，单个聚合 token 已保留大部分动作解码能力。
#### 方法设计详解
第一阶段输入当前帧及四个未来帧，冻结 VAE 和 DINOv3 提取视觉与语义特征，Future Extractor 用查询将其压缩为 32 个、每个 32 维的预测码。预测码作为条件输入冻结视频生成器；视频预测、语义和几何对齐损失共同塑形。生成器中间层经注意力池化形成 32 个世界 token，动作网络同时读取两类表示，动作损失的梯度穿过冻结生成器回传至编码器。
第二阶段输入多视角图像、语言和本体状态。视觉语言主干与世界专家处理当前信息及可学习的未来查询，分别预测第一阶段的两类目标；动作专家通过流匹配生成动作块。训练目标是动作损失加两项蒸馏损失，每项权重 0.1。推理只运行第二阶段，世界专家每次观测运行一次，动作专家迭代去噪十步。
#### 方法对比分析
传统视频 WAM 常部署生成式视觉骨干；这里仅在训练时使用冻结生成器，保留动作相关的预测监督。与只用视觉重建的压缩表示相比，它还让动作梯度塑造码，并利用视频模型内部特征补充世界 token。两个目标分别概括变化和视频先验中的结构，适合对推理成本敏感且有未来轨迹训练数据的操作任务。
#### 实验分析（精简版）
LIBERO-Plus 的 10,030 个扰动回合中成功率为 84.6%，π0.5 为 84.4%；同时移除两类蒸馏降至 75.8%。单 A100 上预测一块动作耗时 25.1 ms，但基线动作长度和实现不同，计时不含环境交互。在 RoboTwin 十个双臂任务的清洁和随机化条件中平均成功率为 93.1%。优势是部署时不运行视频模型，但各方法训练任务数量不同；实验均为仿真，真机效果尚未验证。
#### 实用指南
项目主页为 https://shaolonl-16.github.io/leanwam/，代码和权重的具体开放状态需查项目页。LIBERO 两阶段各训练 30 轮，采用八张 A100；训练成本不含教师缓存预计算。迁移时需匹配相机、动作空间和未来时间跨度，并在目标机器人上验证闭环表现。
#### 总结
核心思想：蒸馏紧凑未来以指导动作。
1. 用未来轨迹构造预测码。
2. 通过冻结视频模型和动作监督塑形。
3. 蒸馏码与世界特征。
4. 仅凭当前观测生成动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.31740v1)
- [arXiv](https://arxiv.org/abs/2611.31740v1)

---

<a id='2611.31901v1'></a>
## [PreWAM: Exploring the Design of Latent Future Prediction for World-Action Modeling](https://arxiv.org/abs/2611.31901v1)

**Authors:** Wan Chi, Sen Li, Zihang Cui, Chuqin Huang, Yunlong Wang, Nanshan Deng, Yanjun Huang

**Published:** 2026-11-24

**Categories:** cs.RO

**Abstract:**

World-action models combine action generation with future prediction, allowing visual transitions to serve as supervision for policy learning. Our controlled study of prediction spaces shows that spatially structured semantic features support better overall generalization under visual perturbations and make motion information more readily recoverable from learned predictive representations. Building on these observations, we introduce PreWAM, a world-action model that organizes future supervision around state changes in a prediction space defined by pretrained DINO features. Specifically, Residual World Prediction jointly learns source-relative changes and their resulting future states to emphasize dynamic information while preserving future-state semantics. Temporal-Scale Prediction conditions world queries on temporal distance to capture both short-term local motion and longer-horizon state evolution. These predictive objectives jointly shape a shared vision-language backbone with action supervision during pretraining and downstream policy learning. Experiments across the LIBERO benchmark family, SimplerEnv, RoboTwin 2.0, and real-world robotic manipulation demonstrate strong manipulation performance and improved generalization and robustness under diverse distribution shifts.

### 论文解读
#### 摘要翻译
世界动作模型把动作生成与未来预测结合起来，让视觉变化也能监督策略学习。作者先比较不同未来表征，发现空间结构化的语义特征在视觉扰动下总体泛化更好，也更容易从预测表示中读出运动信息。基于此提出 PreWAM：在预训练 DINO 特征空间中预测相对当前状态的变化和未来状态，并用不同时间跨度组织预测查询。共享视觉语言主干同时接受未来预测与动作监督。论文在 LIBERO 系列、SimplerEnv、RoboTwin 2.0 和 Piper 实体机器人上评估。

#### 方法动机分析
未来图像既含场景中稳定不变的内容，也含机器人造成的局部变化与外观扰动。只预测像素或单一未来状态，可能把模型容量花在外观细节上，也未明确说明状态如何变化。作者的核心假设是：空间语义特征可保留控制相关结构；同时预测源状态到未来的残差，并覆盖多个时间跨度，能让动作主干获得更直接的运动和时间信息。

#### 方法设计详解
输入当前多视角图像、语言指令和可用本体状态。冻结的 DINOv3 将训练视频帧变成保留空间网格的特征；相邻两帧拼成一个状态，每段视频取四个状态。共享 Qwen3-VL-2B-Instruct 主干处理图像、语言、世界查询与动作查询，分别输出世界预测和动作表示。

Residual World Prediction（RWP）让预测器根据源特征及世界查询输出变化量，再与源特征相加形成预测未来。训练同时用余弦距离对齐预测未来与目标未来、用 Smooth L1 回归目标特征差：前者约束未来语义，后者让变化幅度也受到监督。Temporal-Scale Prediction（TSP）为短、中、长跨度的世界查询加入可学习时距编码，让共享预测器学习局部运动和较长时间的状态演变。动作专家结合动作查询、本体状态和噪声动作序列，以流匹配生成动作块。

预训练交替使用 DROID 机器人示范和 Something-Something V2 人类视频；前者提供动作与未来监督，后者只提供未来预测监督。预训练6,250轮，每轮各含一次机器人更新和视频更新，单次全局批量1,024。LIBERO及SimplerEnv下游阶段使用批量256训练36,000步；RoboTwin 2.0覆盖50任务、训练100,000步。部署只保留视觉语言主干和动作专家，冻结 DINO 编码器与世界预测器不参与推理。

#### 方法对比分析
相较直接预测未来状态，RWP额外预测“变化了什么”；相较单一固定未来间隔，TSP将多个时距显式交给查询。与像素生成型世界模型相比，PreWAM在冻结视觉特征空间做预测，且预测分支只在训练时运行。论文的独特贡献还包括控制变量比较三种预测空间，指出表征选择本身会改变扰动泛化。它适用于视觉操作策略的辅助预训练，不代表可直接替代显式物理仿真或规划器。

#### 实验分析（精简版）
PreWAM在LIBERO-Plus七类扰动上的成功率为79.1%，直接未来状态预测为76.6%，加入RWP后为77.7%；再加TSP提升至79.1%。LIBERO-Long从无TSP的95.0%升至97.0%。在Piper三项任务、每项20次试验中，完整成功率平均61.7%，低于π0.5的68.3%；因此真机结果仍属有限规模验证。不同扰动类别并非都受益，亮度、对比度更明显，噪声和遮挡差距较小。

#### 实用指南
论文报告了主要骨干、特征编码器、批量大小、更新次数和训练数据来源；未明确给出代码或检查点发布链接。复现需重点对齐帧采样、DINO特征配对、RWP双损失、时距分组，以及机器人与人类视频交替比例。推理时不需要预测器，但训练需运行冻结编码器和预测分支。该思路可尝试迁移至其他视觉控制任务，需重新验证时间跨度及传感器扰动是否匹配。

#### 总结
核心思想：在结构化未来特征中联合学变化与时距

1. 用预训练视觉编码器提取带空间布局的未来状态。
2. 通过残差目标与多时距查询监督共享策略主干。
3. 部署时只用主干和动作专家生成动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.31901v1)
- [arXiv](https://arxiv.org/abs/2611.31901v1)

---

<a id='2611.32017v1'></a>
## [Learning Agent States for Latent Action Modeling](https://arxiv.org/abs/2611.32017v1)

**Authors:** Shaoan Xie, Fan Feng, Peiyuan Zhu, Peter Spirtes, Kun Zhang

**Published:** 2026-11-24

**Categories:** cs.CV

**Abstract:**

Latent action models (LAMs) learn action-like latents from unlabeled video by coupling frame-pair inference with latent-conditioned prediction, enabling policy and world-model pretraining. However, what the latent must explain depends on the representation space used for action inference and transition prediction. When this space entangles the agent's evolving physical state with visual context, such as background and persistent agent appearance, inferred latents can encode information unrelated to the agent's actions. To address this challenge, we propose Agent-State LAM, which jointly learns a compact agent-state representation and latent actions modeling its transitions. The compact state aggregates dense visual features into a shared space for both action inference and forward prediction. To guide what this state retains, we introduce spatially matched temporal discrimination (SMTD), which trains the state to distinguish its source frame from nearby frames through spatially flexible matching with frozen local visual features. Rather than comparing global frame representations, SMTD scores each frame by emphasizing its strongest local feature matches across spatial locations. This scoring allows background and persistent agent appearance to remain matched across frames despite displacement, making these shared cues less informative for identifying the source frame and encouraging the representation to capture the agent's evolving physical state. Extensive experiments on distracting control tasks and six real robot video datasets demonstrate improved action recovery, downstream control, and world-model prediction. Code and checkpoints are available at https://github.com/Mid-Push/Agent-State-LAM.

### 论文解读
#### 摘要翻译
论文提出 Agent-State LAM，从无动作标签视频联合学习紧凑的智能体状态和潜动作。通过空间匹配时间判别，模型减少背景和持续外观的干扰，改善动作恢复、控制和世界模型预测。
#### 方法动机分析
潜动作模型需要解释前后帧变化，但变化既可能来自身体动作，也可能来自背景运动或遮挡。压缩潜变量只能限制信息量，无法指定保留内容。作者假设临近帧里背景和外观可跨位置重复匹配，而身体状态更能区分来源时刻。
#### 方法设计详解
输入一对视频帧，共享视觉骨干提取空间特征，注意力池化成紧凑状态。逆动力学模型读取两个状态得到潜动作；前向模型结合当前状态和潜动作预测未来状态，以指数滑动平均编码器的输出为目标。空间匹配时间判别（SMTD）将状态投影到冻结视觉教师的特征空间，与候选帧每个局部补丁计算余弦相似度，以平滑最大值聚合，再用来源帧为正、同视频邻帧为负做分类。能在多个帧中匹配的背景因而难以单独完成判别。训练联合优化预测和判别损失；真实视频实验每个数据集训练步数为 10000，DCS 训练 10 个 epoch；两种设置分别使用不同规模的冻结 DINOv2 特征作为教师。
#### 方法对比分析
与像素重建、全局特征预测或智能体掩码不同，该方法同时限制状态容量并用跨位置的局部匹配塑造状态，不依赖动作标签、掩码或光流。它适合利用无标注视频预训练动作表征，也能为后续策略学习提供潜动作；但判别信号仍依赖场景变化与状态变化之间的统计关系。
#### 实验分析（精简版）
在 DCS hopper 上，动作线性探针误差从 2.70 至 1.14，对照是 LAOM；去掉 SMTD 后为 3.32。使用 32 条标注轨迹训练动作解码器时，归一化控制回报由 0.11 升至 0.37。六个真实机器人视频数据集上的离线动作恢复也改善；在世界模型实验中，模型解码的命令驱动冻结的视频生成器。IRASim 同片生成 PSNR 为 23.35 dB，高于次优 CoMo 的 22.63 dB；但 RT-1 轨迹准确率由光流目标法领先（56.4 对 53.3）。旋转和夹爪命令仍难恢复；真实视频实验未证明实体机器人闭环效果。
#### 实用指南
代码与权重见 [GitHub](https://github.com/Mid-Push/Agent-State-LAM)。复现时保持同视频邻帧采样、冻结 DINOv2 局部特征和损失梯度比例；从潜动作得到可执行命令仍需有标签解码器。迁移到驾驶时应保留其他交通参与者的相关运动。
#### 总结
核心思想：用时空匹配学习智能体状态。
1. 提取并池化帧特征，得到紧凑的状态表示。
2. 从状态对推断潜动作，并预测未来状态。
3. 用邻帧局部匹配塑造状态，使重复外观更难成为判别捷径。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.32017v1)
- [arXiv](https://arxiv.org/abs/2611.32017v1)

---

<a id='2611.32198v1'></a>
## [MoGe-4D: Efficient, Flexible, and Unified Dense 4D Reconstruction and Tracking](https://arxiv.org/abs/2611.32198v1)

**Authors:** Ruicheng Wang, Linlin Xin, Sicheng Xu, Jianfeng Xiang, Zelong Lv, Yu Deng, Yue Dong, Guangzhong Sun, Tong Zhang, Jiaolong Yang

**Published:** 2026-11-24

**Categories:** cs.CV

**Abstract:**

Reconstructing dynamic scenes from RGB video requires reasoning about both 3D geometry and motion across viewpoints, yet existing methods either split reconstruction and tracking into fragmented pipelines or face prohibitive dense spatiotemporal decoding costs. We introduce MoGe-4D, a feed-forward framework that unifies geometry and motion prediction via ordered queries of paired allocentric and egocentric dynamic point maps. Each query maps source pixels to a target time simultaneously in both camera frames, resolving relative camera pose in closed form while flexibly serving depth, dynamic registration, and dense 3D tracking. For efficient dense querying, MoGe-4D decouples scene modeling from query decoding: a high-capacity context encoder caches coarse global spatiotemporal features once, while a lightweight query encoder rapidly decodes dense point maps at flexible resolutions. Furthermore, as supervising dense 4D queries demands continuous 3D correspondences whose raw trajectory storage is prohibitive, we introduce RigIDTrack, a synthetic data pipeline and dataset that compresses surface dynamics into rigidity-regularized affine bodies, recovering exact arbitrary-frame 3D trajectories with $\sim15\%$ storage overhead over RGB-D video. Across diverse benchmarks, MoGe-4D achieves state-of-the-art dense 3D tracking, matches dedicated 3D foundation models in reconstruction, and runs substantially faster than prior 4D baselines.

### 论文解读
#### 摘要翻译
本文提出 MoGe-4D，从无位姿 RGB 视频统一预测动态场景三维几何、相机运动和稠密点跟踪。它以缓存全局上下文、轻量按需查询的双编码器降低逐点解码成本；另提出 RigIDTrack，用仿射体压缩连续表面运动轨迹。

#### 方法动机分析
动态重建需联合几何、相机和物体运动。分阶段估计容易割裂任务，逐点查询则难扩展到密集视频。作者假设，同一源像素在目标时刻的源、目标相机坐标点图可互补描述运动并提供相机对应；全局特征先计算复用，细节按查询解码。

#### 方法设计详解
输入是无位姿 RGB 序列和源帧—目标时刻查询 (i,j)。模型输出源坐标 allocentric 点图与目标坐标 egocentric 点图，分别利于稳定背景和运动前景。加权对齐两图可闭式求相机相对位姿；自查询估焦距与深度偏移，共视性构建稀疏位姿图。动态掩码融合世界坐标点图后，多源帧重建目标时刻，固定源帧遍历时间则得到稠密三维轨迹。

上下文 ViT 在粗网格编码整段视频并缓存多层特征；查询编码器按需重编码源图，以条件注意力读取源、目标上下文，由卷积头输出双点图及不确定度、可见性和动态掩码。MLP 预测尺度与共视性。默认每帧 500 个上下文 token、900 个点图 token；训练最多 32 帧，使用 64 张 A100-40GB、全局 batch 128，训练 160k 步。
RigIDTrack 将运动一致的表面片元合并为共享仿射体，按帧保存 RGB-D、相机参数、像素体 ID 和体变换，以矩阵链恢复任意时刻轨迹，避免逐像素保存长序列。

#### 方法对比分析
核心区别在于它以双坐标点图贯通重建、跟踪和位姿估计，并以局部帧对避免依赖单一参考帧。相较分离几何与光流的架构，它复用全局特征，但需预测并融合双地图及辅助掩码，适合动态视频几何与稠密跟踪。

#### 实验分析（精简版）
六个真实与合成跟踪基准的 32 帧测试中，V-JEPA 世界坐标全点 RelEPE 为 0.077，V-DPM 为 0.138；AUC25 分别为 78.89% 和 72.65%。六个重建基准上，其全局点 AUC25 为 80.86%，接近 π3 的 81.09%。50 帧稠密跟踪约比 4RC 快 4 倍。边界与细结构易模糊，V-JEPA 超过约 100 帧退化；Open-D4RT 速度比较采用未使用查询去重的复现。

#### 实用指南
论文提供项目页，Open-D4RT 基准为非官方复现。默认 500/900 token 可作为起点，按自查询估焦距并用共视性建位姿图。训练使用 64 张 40GB A100。机器人场景跟踪可尝试使用世界坐标轨迹，但需验证传感器域差异与实时约束。

#### 总结
核心思想：双坐标点图统合动态四维感知。
1. 编码视频一次，缓存粗粒度时空上下文。
2. 输入源帧与目标时刻，按需解码双坐标点图。
3. 从对应点求相机位姿并融合到世界坐标。
4. 改变帧对查询，即可重建场景或追踪稠密轨迹。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.32198v1)
- [arXiv](https://arxiv.org/abs/2611.32198v1)

---

