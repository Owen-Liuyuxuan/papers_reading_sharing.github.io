time: 20261022

# Arxiv Computer Vision Papers - 2026-10-22

## Executive Summary

## 今日研究脉络

本日筛选出的十篇论文主要围绕机器人从仿真走向真实部署时的三个瓶颈：真实感感知与高吞吐仿真、规划控制中的可靠性，以及部署过程本身的在线学习。多篇工作把几何、物理或结构化先验放入学习闭环，避免仅靠更大模型或更多随机数据掩盖系统误差。

## 代表性进展

《PRISM》分别建模深度传感器的无效区域与测量误差，在六项真实操作任务中将RISE平均成功率提升至92.5%，高于Stable-Sim2Real的90.8%；但其训练仍依赖特定相机数据。《COWBOY》把场景级全身控制与局部抓取专家蒸馏为部分观测下的统一移动操作策略，在24个真实场景取得91%的平均抓取成功率，展示了系统级仿真训练与真实场景覆盖的结合。

系统基础设施也有明显推进。《GGraph》通过GPU张量化导航图提升并行探索环境更新吞吐，并在相同训练预算下取得有竞争力的路径长度；《HyperDrone》将飞行动力学与光线追踪渲染统一加速，完成视觉偏航控制的实机迁移概念验证。两者提醒读者，数据生成和地图更新速度会直接影响可探索的训练规模，但吞吐提升仍需由任务指标验证。

可靠性方向从不同层次切入：《SLoSa》以形式验证约束小型语言模型的符号计划；《Distance Field Driven Reinforcement Learning》把全身距离梯度加入策略观测；《A2I-Calib++》设计抗噪运动并使用腿部运动学改善多IMU标定。另有《Linear Twisting Sidewinding Gaits》从机械执行器负载分配优化蛇形机器人步态，体现了学习方法之外的机构与控制设计价值。

## 值得关注的方向

《Training in Deployment》把人类接管与策略动作按来源记录，并在部署会话中在线更新world-action模型；《Learning to Absorb Corrections》则让成功纠正同时改进动作先验与残差策略。两者都把部署数据纳入学习，但现有评估仍依赖有限平台、任务和人工监督。读者可重点比较：系统是否提供可验证的安全边界、实机收益是否来自策略本身，以及成功判据是否覆盖长时任务。

## 阅读建议

关注感知迁移与Sim-to-Real的读者可优先阅读PRISM、COWBOY和HyperDrone；研究机器人安全规划、避障和状态估计的读者可比较SLoSa、距离场方法与A2I-Calib++；在线学习和纠正吸收方向则可从TIDE与ACT-Flow入手。阅读时应同时核对真实实验条件、比较基线和论文明确承认的适用边界。

---

## Table of Contents

1. [Linear Twisting Sidewinding Gaits for Twistable Snake Robot Motion Planning on Complex Terrains](#2610.29461v1)
2. [SLoSa: Small, Low-Power, and Safe Foundation Models for Robotic Task Planning](#2610.28955v1)
3. [PRISM: Learning Realistic Depth via Physics-Grounded Noise Disentanglement with Semantic-Geometric Collaboration](#2610.29432v1)
4. [GGraph: A GPU-Accelerated Library for Graph-Based Robot Deployment and Learning](#2610.28953v1)
5. [HyperDrone: Simulation-to-Reality Transfer at Millions of Frames per Second](#2610.28950v1)
6. [COWBOY: A Scalable Sim-to-Real Framework for Learning Contextual Whole-Body Manipulation](#2610.28948v1)
7. [Training in Deployment: Human-in-the-loop Online Post-Training for Robot Manipulation](#2610.29264v1)
8. [Learning to Absorb Corrections: Critic-Weighted Residual Flow for Vision-Based Insertion](#2610.29117v1)
9. [Distance Field Driven Reinforcement Learning for Reactive and Robust Collision Avoidance](#2610.29452v1)
10. [A2I-Calib++: A Kinematics-Driven Anti-noise Active Calibration Framework for Whole-Body Multi-IMU Systems on Legged Robots](#2610.29438v1)

---

## Papers

<a id='2610.29461v1'></a>
## [Linear Twisting Sidewinding Gaits for Twistable Snake Robot Motion Planning on Complex Terrains](https://arxiv.org/abs/2610.29461v1)

**Authors:** Qingling Fan, Zhongqiang Ren

**Published:** 2026-10-21

**Categories:** cs.RO

**Abstract:**

Sidewinding is an efficient gait for snake robots to locomote on various terrains, by undulating the backbone curve and shifting the contact segments along it. However, relying solely on undulatory motion limits the locomotive capability of snake robots, and combining it with twisting motion can improve their speed and robustness. This hypothesis has recently been verified by the rolling vector approach, which locomotes the robot like a differential-drive car. Yet its twisting is concentrated in a short buffer segment, imposing most of the workload on the two twist joints in the middle, which can often overburden the motors and limit its locomotion. With that in mind, this paper develops a new family of twisting sidewinding gaits that distribute the workload over the entire body with our proposed linear twisting function. Under matched turning speed and total output power, our gaits reduce the peak output power of a single twist joint by 42%. Real-robot experiments show that, with the reduced workload, our new gaits can locomote the robot in various terrains that the existing rolling vector approach cannot. Besides, our new gaits can move up to 49% faster than conventional sidewinding without twisting.

### 论文解读

#### 摘要翻译

作者提出一种面向可扭转蛇形机器人的线性扭转侧向步态，让机器在复杂接触地面上前进和转向时，将身体扭转分配到更长的躯干范围。论文研究的是明确的机构与运动控制问题，重点不在通用机器人策略。

#### 方法动机分析

传统滚转向量方法把扭转载荷集中在中间缓冲段，少数关节先达到功率或电机限制，整体速度因此受限。作者假设让扭转沿身体近似线性变化，就能在维持相近运动效果时摊平关节负载，并保留侧向步态的可解释参数。

这一设定把轨迹生成和执行器负载放在同一问题中考量。其关键假设是关节可按连续参考曲线同步动作，因而比较时必须保持任务和总功率公平。

#### 方法设计详解

方法先用弧长和时间参数化扭转相位，再由该相位计算蛇体背侧参考曲线及所需侧向弯曲。连续曲线随后按离散关节分段近似积分，形成硬件可执行的关节轨迹。参数中的空间项控制头尾扭转差异，时间项控制整体扭转速度，因此可以把平移与不同转向模式放进同一表达。

实际实现先定义线性相位，再计算背侧参考曲线与关节角；离散模块的分段积分近似连续解，随后对六个扭转关节下发周期轨迹。参数调节后可分别形成平移、尾部锚定转弯、原地转弯和头部锚定转弯。

#### 方法对比分析

与集中扭转的滚转向量方案相比，这项工作的主要贡献是把扭转作为沿身体分布的自由度来设计，而不是只改变整体滚转速度。其适用对象是具有独立扭转关节、可执行侧向波形的蛇形机器人；方法是否适合其他关节布局，需要重新建立机构运动学映射。

#### 实验分析（精简版）

作者先匹配两种步态的转向角速度和总功率，再比较关节功率分布；基线96.8%的扭转功率落在中间两关节，新方案最重的两个关节合计45.1%，最大单关节功率从9.89 W降至5.76 W。实机在石板、草地、碎石和20度坡面各进行五次、每次2米试验，到达时间分别缩短45.1%、41.0%、47.1%和49.2%。测试证明了该平台上的效果，但地形和样本范围仍有限。

结果来自作者的实体T-Snake，而非仅有仿真轨迹；地形计时反映整体通行速度，功率比较则说明其执行器负担变化。两类测量共同支撑作者关于速度与负载改善的主张。

#### 实用指南

复现时应关注机器人各模块的扭转自由度、关节输出功率定义、分段积分精度，以及对比时是否匹配角速度与总功率。论文使用T-Snake实体平台，并公开讨论参数设置与未来整合非侧向步态的方向；迁移到其他蛇形机构前，需重新标定关节执行器并验证接触摩擦和负载耦合。

代码和实验平台细节应以论文提供内容为准；实施前还需核对关节功率测量口径、机械限位及地面摩擦，避免将理想曲线直接用作硬件命令。

#### 总结

核心思想是让扭转沿蛇身均匀分担。
1. 设定随弧长和时间变化的扭转相位。
2. 从参考曲线计算离散关节步态。
3. 在相同运动条件下比较关节负载和地形通行时间。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.29461v1)
- [arXiv](https://arxiv.org/abs/2610.29461v1)

---

<a id='2610.28955v1'></a>
## [SLoSa: Small, Low-Power, and Safe Foundation Models for Robotic Task Planning](https://arxiv.org/abs/2610.28955v1)

**Authors:** Weizhe Xu, Jialiang Fan, Essien Taylor, Zijun Wang, Russell Joseph, Fanxin Kong

**Published:** 2026-10-20

**Categories:** cs.RO, cs.AI, cs.LO

**Abstract:**

Safe robotic task planning requires planners that can reliably achieve task goals while satisfying safety constraints. Classical planners search over explicit symbolic task representations and can provide formal guarantees with respect to encoded safety constraints, but require rigid input formats. Their computational cost also increases rapidly with the search space. In contrast, large language models (LLMs) offer flexible task interfaces and strong generalization capabilities, but they are large, power-hungry, and pre-trained with little robot-specific safety data and knowledge. Thus, they can be difficult to deploy locally on robotic systems with limited resources. To address these limitations, we present SLoSa, a neuro-symbolic compression and fine-tuning framework for training foundation-model-based robotic task planners with Small size, Low power demand, and Safety awareness. The framework first constructs a safety-aware dataset using classical planners, an advanced teacher LLM, and formal verifiers, followed by two training stages: supervised fine-tuning and reinforcement learning (RL) fine-tuning guided by a verification-based reward machine to enforce safety alignment. We evaluate SLoSa across four Planning Domain Definition Language (PDDL) domains, two household simulation environments, local edge platforms, and a desktop robotic arm. Across the PDDL tasks, verifier-guided post-training raises the success rate of compact 4B-14B planners to 70.5-76.0%, while INT4 deployment on a Jetson Orin requires 17.85-26.41 W of average power. Results in household simulators and on the physical robot further demonstrate the applicability of SLoSa across diverse robotic task settings.

### 论文解读

#### 摘要翻译

论文提出SLoSa，一套面向机器人任务规划的小模型训练与验证流程。它的目标是在算力有限的设备上运行轻量语言模型，同时让规划结果遵守可检查的任务约束，并在实际执行前经过筛选。

#### 方法动机分析

直接用语言模型生成计划会出现遗漏前置条件、错误操作顺序或违反新增安全要求的问题；仅靠自然语言示例也难以稳定覆盖约束边界。作者的核心假设是：如果训练样本包含经过形式验证、且约束确实改变计划结果的例子，模型就能学会把安全条件融入符号计划。

这项假设把“计划可验证”作为安全边界，但并不意味着真实机器人不会碰撞或操作失败；因此论文将高层符号约束与低层执行结果分开报告。

#### 方法设计详解

流程首先从经典PDDL任务构造安全约束变体，再由传统规划器和教师模型提出候选计划。形式化验证器检查候选是否满足动作前提、顺序和约束，并把有效样本用于小型量化模型的监督微调。随后通过验证器反馈进行GRPO强化，使模型在生成阶段更偏向可验证计划。部署时还会再次验证，只将通过的高层计划映射到预定义机器人技能。

在训练中，规划器产生可行解，教师模型提供候选表达，验证器检查硬约束并给出反馈。通过筛查的数据用于监督微调，之后GRPO利用验证器结果优化策略；部署端仍需在执行前做一次验证。

#### 方法对比分析

不同于把语言模型直接接入控制器，SLoSa把模型限制在高层符号规划，并把规划器与验证器放在数据准备和执行边界上。它与单纯监督微调的区别在于加入验证器引导的策略优化；安全结论也明确局限于符号任务约束，不能替代碰撞检测、运动规划或执行监控。

#### 实验分析（精简版）

论文在四个PDDL领域及VirtualHome、ALFWorld-Safe等任务上比较了模型与训练方法。Jetson Orin部署的平均功耗为17.85–26.41瓦。桌面机械臂30次试验中，SFT有13次成功，GRPO有14次；其中一些已通过符号验证的计划仍因堆叠精度或工作空间限制而未完成，说明形式正确和实体成功不是同一指标。

仿真基准用于观察任务成功和安全约束满足情况，真实桌面臂试验则把计划接受和动作完成分列。这样的分层指标很重要，因为验证器只知道符号状态模型，无法完全表示抓取误差、机械臂奇异点或物体滑动。

#### 实用指南

复现时要保留约束变体生成、规划器标签、验证器规则和执行技能映射，不能只复制模型训练配方。论文提供PDDL基准和边缘设备测量设定；迁移到新机器人时，需要补齐其动作语义、可达性检查及真实执行失败处理。

论文的PDDL任务与执行技能映射是复现重点；将方法用于新的机器人前，应先验证动作模型是否覆盖真实前置条件，并明确验证失败后的重试策略。

#### 总结

核心思想是让可验证约束进入训练闭环。
1. 构造会改变计划的安全约束。
2. 用规划器与验证器筛选训练数据。
3. 训练小模型并在执行前再次校验。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.28955v1)
- [arXiv](https://arxiv.org/abs/2610.28955v1)

---

<a id='2610.29432v1'></a>
## [PRISM: Learning Realistic Depth via Physics-Grounded Noise Disentanglement with Semantic-Geometric Collaboration](https://arxiv.org/abs/2610.29432v1)

**Authors:** Xiujian Liang, Jiacheng Liu, Mingyang Sun, Qichen He, Anda Cheng, Cewu Lu, Jianhua Sun

**Published:** 2026-10-21

**Categories:** cs.RO

**Abstract:**

Real-world physical sensing exhibits complex, heterogeneous noise patterns that deviate significantly from idealized simulation, posing a fundamental bottleneck for sim-to-real transfer. Existing sensor modelings typically treat depth noise as a monolithic black-box process, overlooking the distinct physical mechanisms that govern different error modalities. In this work, we introduce a physics-grounded paradigm that disentangles monolithic noise into two complementary modalities: \textbf{sensing invalidation} and \textbf{measurement inaccuracy}, enabling a tailored treatment of noise sources based on their physical origins. Building on this insight, we propose \textbf{PRISM}, a tripartite framework that distills 3D Visual Foundation Model features as rich spatial-semantic priors for physics-based reasoning. To address the inherent sparsity and class imbalance of invalidation regions, we develop Hierarchical Positive-Prioritized Supervision, integrating multi-scale positive-weighted objectives with a positive-preserving dynamic hard mining strategy to enforce precise artifact delineation. Extensive benchmarks demonstrate that PRISM achieves state-of-the-art fidelity in noisy depth synthesis. Furthermore, downstream robotic experiments show that PRISM facilitates a \textbf{92.1\%} average success rate in the real world, marking a significant improvement over monolithic baselines. Our project page is available at: \resizebox{\linewidth}{!}{\href{https://shawnleung-sii.github.io/prism-icml2026/}{https://shawnleung-sii.github.io/prism-icml2026/}.}

### 论文解读

#### 摘要翻译

PRISM是一种面向机器人Sim-to-Real的深度图合成方法。它试图让仿真深度更像真实传感器在不同材质、反射和遮挡条件下产生的数据，从而让只在仿真训练的策略面对真实深度异常时仍能工作。

#### 方法动机分析

常见做法把真实深度噪声当成整体随机扰动或图像风格差异，难以区分传感器根本测不到的区域与虽有偏差但仍有效的测量。作者认为先识别材质和几何，再显式区分无效像素与测量误差，可以让生成过程更接近深度相机的物理成因。

其核心假设是材质和局部几何能够预测深度传感器的失效模式，因此错误掩码与连续误差分别学习。该设计减少将不同物理现象混为一种噪声的风险。

#### 方法设计详解

输入包括模拟RGB和深度。冻结的三维视觉基础模型提供语义与几何先验；BND模块根据这些线索预测传感无效掩码；扩散式残差生成器在语义几何条件下产生测量误差，再与模拟深度融合。针对无效像素少、边缘难学的问题，H-PPS组合多尺度正类加权损失和逐渐增加的困难负例挖掘。训练使用8张H200、100轮、512×512图像及32的批量，生成采用50步DDIM。

从输入到输出的步骤是：先由冻结的三维模型提取几何语义，再估计无效像素掩码；随后用条件扩散模型预测有效深度上的残差，最后合成带真实感噪声的深度图供策略训练。训练批量为32，历时100个epoch，图像尺寸512乘512。

#### 方法对比分析

与直接用噪声扰动或传统仿真深度增强不同，PRISM把无效区域和有效测量偏差分别建模，并尝试跨策略复用同一数据生成模块。它适合深度相机感知差异是主要Sim-to-Real瓶颈的任务；模型依赖具体传感器数据，不能视为不需适配的通用相机模拟器。

相较于仅做风格迁移或随机扰动，这一贡献是依据物理语义分别处理失效与有效测量误差。

#### 实验分析（精简版）

下游实验涵盖Franka与Realman平台上的六项抓取、放置和长程操作任务。使用RISE策略时，PRISM的平均成功率为92.5%，Stable-Sim2Real为90.8%，仅用干净仿真为43.1%；跨三类策略结构的平均值为92.1%对89.1%。这些结果支持其在测试传感器和任务上的效果，但论文也指出逐帧生成没有时间一致性，且未模拟交互造成的外参变化。

比较同时覆盖深度重建质量和真实机器人任务成功率，并用去除SPR、BND的消融判断各模块作用。下游提升说明合成观测有用，但并不能说明深度值逐帧正确，也未回答长时间视频中的抖动问题。

#### 实用指南

复现需要真实深度训练数据、相机参数及统一的无效像素定义；计算成本也较高，论文训练采用多张H200。迁移到不同相机时应重新评估无效区域分布和深度量程；在机器人抓取之外的长时感知任务中，还需检查逐帧闪烁是否影响策略。

迁移时需采集目标相机数据，匹配分辨率、视场与无效像素形态，并重新标定几何先验；新的相机未经评估时不宜沿用原模型结果。

#### 总结

核心思想是按物理成因合成深度噪声。
1. 由RGB与几何先验判断材质。
2. 区分无效区域和有效测量误差。
3. 合成真实感深度并训练下游策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.29432v1)
- [arXiv](https://arxiv.org/abs/2610.29432v1)

---

<a id='2610.28953v1'></a>
## [GGraph: A GPU-Accelerated Library for Graph-Based Robot Deployment and Learning](https://arxiv.org/abs/2610.28953v1)

**Authors:** Yash Rampuria, Rohan Garg, Daniele Gammelli, Marco Pavone

**Published:** 2026-10-20

**Categories:** cs.RO

**Abstract:**

Autonomous mobile robots require spatial representations that capture geometry and semantics while staying sparse enough for planning in large, unknown environments. Graphs provide a natural representation for this purpose and underpin both classical planning methods and a growing class of reinforcement-learning (RL) policies. However, RL training needs hundreds to thousands of parallel environments, each constructing and updating its own graph. Existing graph libraries are typically designed for either single-robot deployment or fixed graphs, and CPU-based graph construction limits scalable training to only a few dozen environments. We introduce GGraph, an open-source GPU-accelerated graph library for both robot deployment and large-scale learning. GGraph supports real robots through ROS 2 interfaces and vectorized GPU-based policy training, with built-in support for tasks including 3-D navigation, exploration, and loop-closure integration. For single-robot operation, GGraph updates graphs in milliseconds. At scale, it enables graph-based exploration policies to train across hundreds of parallel robots, reducing wall-clock training time by one order of magnitude. Across simulation and real-world experiments, these results establish GGraph as a unified graph infrastructure for both large-scale robot learning and deployment.

### 论文解读

#### 摘要翻译

GGraph是一套用GPU加速图式机器人部署与学习的库，面向探索和导航中反复建图、更新、搜索带来的计算开销。它把图计算放到与机器人策略相同的GPU执行路径上，目标是增加可并行环境数量并缩短策略训练时间。

#### 方法动机分析

图式探索常需将点云转成占据网格和导航图，再执行可见性、路径代价和闭环更新。若地图规模大、并行机器人多，CPU侧图运算和数据搬运会成为瓶颈。作者的出发点是把常用图表示和操作做成持久化GPU张量，让环境更新与策略训练批量运行。

作者的核心假设是图更新与查询可以变成批量GPU张量运算，从而与并行仿真同频，而非仅把单次路径搜索做快。

#### 方法设计详解

系统从局部点云建立局部导航图，随后合并进全局图并对过期、冗余区域进行裁剪。节点还保存探索、导航所需属性；图查询包括视线检查、广度优先和Bellman–Ford等算法，闭环模块修正漂移轨迹。各步骤由CUDA并行实现。作者报告100K节点图更新约7毫秒，1M节点约62毫秒；在512个并行环境下，完整环境更新吞吐达到约6500步/秒。

流程从局部点云生成局部图，随后持久合并到全局GPU图，更新节点属性并裁剪失效区域；学习器通过统一接口调用视线检测、图遍历、最短路和闭环修正。吞吐对比在RTX 4090上使用固定转移预算，并改变并行环境配置。

#### 方法对比分析

GGraph的创新主要是高吞吐图计算系统，而不是一种新奖励函数或新策略。与依赖CPU图结构的导航流程相比，它强调GPU张量化、批量更新和与强化学习环境紧耦合。对图搜索占主要开销的探索任务较有价值；若任务瓶颈来自视觉推理、通信或控制，图运算加速未必转化为端到端收益。

#### 实验分析（精简版）

在150张验证地图上，相同6小时训练预算的GGraph-PPO平均路径长度为1255像素、成功率97%；对比使用发布模型且训练24小时的ARiADNE策略，分别为1333像素和91%。另一种GGraph策略在容易地图成功率更高，说明路径质量与任务完成率存在取舍。论文也展示Spot真实室内运行，但数据量较小，主要证明可部署性。

实验既报告节点规模下的图更新时间，也用等训练时长的策略评估地图覆盖和行走路径，因此系统速度与下游效果都有对应测量。高吞吐本身不等于所有导航任务路径更优。

#### 实用指南

复现时需统一图节点定义、地图分辨率、更新窗口及环境步数统计，否则吞吐对比没有可比性。作者测试包括并行仿真、户外模拟和实体Spot。向自动驾驶或大规模仓储迁移时，应先确认地图拓扑更新是否真是主耗时，再评估动态障碍和多机器人通信。

复现时需要报告地图构建、节点规模、环境数、步数预算和GPU型号。对于非图式或地图频繁变化的任务，应先做端到端剖析再决定是否使用该库。

#### 总结

核心思想是把导航图计算搬到GPU。
1. 将点云转换为局部图。
2. GPU合并、更新并查询全局图。
3. 并行训练探索策略并评估部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.28953v1)
- [arXiv](https://arxiv.org/abs/2610.28953v1)

---

<a id='2610.28950v1'></a>
## [HyperDrone: Simulation-to-Reality Transfer at Millions of Frames per Second](https://arxiv.org/abs/2610.28950v1)

**Authors:** Jonas Eschmann, Giuseppe Loianno

**Published:** 2026-10-20

**Categories:** cs.RO, cs.AI, cs.LG, eess.SY

**Abstract:**

We present HyperDrone, an extremely fast quadrotor simulator focused on high-performance batch rendering that is more than 20x faster than existing state-of-the-art batch quadrotor simulators. HyperDrone combines highly optimized multirotor dynamics with HyperLux, a new ray-tracing engine that supports Nvidia OptiX, Vulkan, Apple Metal and WebGPU backends that takes full advantage of modern ray-tracing hardware, capable of running both components entirely on the GPU. HyperLux renders RGB, depth, semantic/instance segmentation, surface normals and optical flow observations at configurable levels of fidelity, including reflections, realistic lighting, motion blur, anti-aliasing, and Physically Based Rendering (PBR). Even in the highest-fidelity settings, HyperLux achieves millions of Frames Per Second (FPS) on consumer-grade GPUs while rendering realistic 3D scenes. We demonstrate the merits of HyperDrone by training a vision-based yaw-control policy purely in simulation and transferring it to a real small-scale quadrotor (Crazyflie) with a camera, running completely onboard.

### 论文解读

#### 摘要翻译

HyperDrone是一套面向四旋翼机器人的高速GPU模拟器，目标是在保持图像真实性的同时大幅提高并行训练数据生成速度。论文将动力学和光线追踪渲染整合，支持从合成图像训练视觉策略，再将简化控制策略迁移至真实无人机。

#### 方法动机分析

视觉机器人学习需要大量环境交互，但高保真渲染常比动力学计算更慢；渲染太简化又会扩大仿真与现实的差异。作者的思路是利用GPU并行动力学与专用光线追踪后端，使模拟器能在大量环境中快速重渲染，同时提供丰富的传感器输出。

作者采用可程序化室内场景以检验不同布局的视觉多样性，但仿真数据分布仍可能留下系统性偏差；高速采样不能自动保证域随机化充分。

研究动机是降低渲染成为大规模飞行学习瓶颈的影响；关键假设是GPU光追能同时保留场景细节并支持批量采样。

#### 方法设计详解

系统一侧运行四旋翼动力学，另一侧由HyperLux根据三维场景产生RGB、深度、语义和实例分割、表面法线及光流等观测。渲染支持OptiX、Vulkan和Metal；不具备专用光线追踪硬件时，可用WebGPU软件遍历。作者强调每帧都可重新渲染，减少重复样本，但仍可能对模拟器生成的数据分布过拟合。实机迁移实验让网络读取当前图像和目标图像，估计相对偏航角，由已有低层控制器完成其余飞行控制。

训练流水线在场景网格中采样无人机状态，动力学给出位姿，渲染器再以相机和几何交点输出像素。学生网络仅预测目标相对偏航角，低层姿态稳定与电机命令由RAPTOR负责，因此这是视觉迁移的受控验证。

#### 方法对比分析

与只提供简化画面的飞行模拟器相比，HyperDrone把多模态渲染和动力学放入一个批量GPU流水线；与端到端飞行研究相比，其迁移实验刻意只学习视觉偏航控制。后者减少了控制因素干扰，适合研究视觉域差异，但也限制了可据此作出的自主飞行结论。

#### 实验分析（精简版）

在RTX 4090、4096个环境、64×64图像条件下，作者报告其吞吐超过最快对比四旋翼模拟器20倍。真实飞行实验中，视觉学生训练约4小时后能够跟踪目标。部署使用OpenMV AE3摄像头板上的NPU，RAPTOR以500Hz执行低层控制，位置还依赖外部参考；论文未实现集成式避障。

吞吐基准统一图像尺寸与并行环境数比较模拟器；实机结果则检验目标跟踪能否在真实相机画面闭环。两者回答不同问题，前者衡量训练数据产能，后者证明有限控制任务可迁移。

#### 实用指南

复现时需固定GPU型号、环境数、图像尺寸和渲染保真度，吞吐数字才可比较。实机实验还依赖相机安装、偏航数据和外部定位。迁移到更复杂无人机任务时，应加入滚动快门、碰撞规避和自主位置估计等环节，并单独评估它们的失效模式。

复现需固定渲染保真档位、显卡和环境批量，实机还要标定相机视场与偏航命令比例。对高动态飞行还应加入滚动快门和运动模糊变化验证。

#### 总结

核心思想是用GPU把真实感飞行数据批量生成。
1. 并行模拟四旋翼动力学。
2. 光线追踪生成多模态观测。
3. 训练视觉策略并在受控实机任务验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.28950v1)
- [arXiv](https://arxiv.org/abs/2610.28950v1)

---

<a id='2610.28948v1'></a>
## [COWBOY: A Scalable Sim-to-Real Framework for Learning Contextual Whole-Body Manipulation](https://arxiv.org/abs/2610.28948v1)

**Authors:** Jiahui Yang, Jason Jingzhou Liu, Ray Muxin Liu, Andrew E. Wang, Kenneth Shaw, Deepak Pathak, Ruslan Salakhutdinov

**Published:** 2026-10-20

**Categories:** cs.RO

**Abstract:**

Mobile manipulators must adapt not only to objects, but also to the spatial context in which they appear: where to position the base, how to manipulate, and where to look while navigating surrounding scene geometry. To address this, we present COWBOY, a scalable sim-to-real framework for learning such contextual whole-body manipulation. On whole-body dexterous grasping, COWBOY transfers zero-shot to hundreds of objects and diverse in-the-wild scenes, spanning cluttered tabletops, shelves, drawers, and other unstructured settings. Rather than learning scene adaptation directly with RL in simulation, a vectorized environment-aware whole-body controller (WBC) handles reaching in thousands of scenes by coordinating base-arm motion, collision avoidance, and active vision. Then, local RL experts are trained in local scenes for object interaction and can be reused across thousands of full-scene layouts. We distill these composed WBC-RL teachers into a single vision policy that controls the mobile base, arm, dexterous hand, and actuated camera from point-cloud observations. The resulting policy exhibits contextual whole-body behavior, automatically adapting base placement, camera viewpoint, arm trajectory, and grasp strategy to observed scene and object geometry. Videos are available at https://cowboy-manip.com

### 论文解读

#### 摘要翻译

COWBOY研究移动操作机器人如何在杂乱、未见过的真实场景中选择接近位置、调整身体姿态并抓起目标。作者提出从大量仿真中的场景级教师策略蒸馏出一个只需部分点云观测的统一策略，力图让移动底盘、手臂、灵巧手和相机协同工作。

#### 方法动机分析

灵巧抓取策略常局限在物体附近，无法独自解决远距离接近、避障和视角调整；整体端到端强化学习又很难同时学好场景导航与精细操作。作者的假设是把问题拆成全身控制器负责场景尺度运动、局部强化学习专家负责物体交互，再通过教师学生蒸馏把特权信息转化为真实可获得的视觉观测。

其核心假设是局部抓取能力可由多个专家高效学习，而全身控制器能够把机器人送入这些专家的有效状态分布；蒸馏再消除教师的全场景特权信息。

#### 方法设计详解

教师阶段先生成包含目标和周围障碍的完整场景。fabrics全身控制器让机器人接近局部交互区域、调整底盘并保持注视，局部PPO专家执行不同抓取方式。学生用DAgger从教师示范学习，输入相机与LiDAR点云、机器人状态和目标估计，由点云条件Transformer输出底盘速度及手臂、手和颈部动作。训练包含8192个并行仿真环境、250种物体和8张A6000 GPU，耗时约60小时。

输入首先包括目标估计、机器人状态和点云，教师在完整仿真状态下选择接近和抓取行为，学生通过DAgger模仿教师并逐渐适应自己访问的状态。训练使用8192个并行环境和8张A6000 GPU，实际部署读取相机及LiDAR观测。

训练设置还包括约60小时仿真训练，部署输入为相机与LiDAR点云；网络控制底盘、手臂、手和活动相机，不直接输出关节力矩。

#### 方法对比分析

与分别训练每个物体的抓取专家相比，COWBOY先学场景关系和全身动作，再把多位教师压缩成单个视觉策略。它强调在同一个策略中联合底盘位置、视角、碰撞规避与抓取方式，适合给定目标物后的移动抓取；论文没有解决从语言意图中发现目标或规划完整任务链。

#### 实验分析（精简版）

真实评估覆盖24种未在训练中建模的室内场景，平均成功率91%；仿真留出环境中，学生92.4%，教师95.6%。作者也报告纯端到端RL基线在相同仿真设置下为0%。实机结果覆盖桌面、抽屉、货架、椅子、冰箱等场景，但每种物体仍使用目标分割初始化，不能据此推断完全开放式物体发现能力。

实机评测按场景和物体分组，并将抬升至少10厘米且无碰撞定义为成功；仿真留出结果单独衡量蒸馏差距。这些定义让91%和92.4%各自对应清楚条件。

#### 实用指南

复现需要控制器、局部专家、场景生成、传感器渲染和DAgger数据链同时工作；仅复刻策略网络不会得到同等系统。部署时SAM3提供目标初始分割，颈部相机和LiDAR对主动视觉很重要。迁移到仓储或家庭移动操作前，还要补充目标语义选择、任务完成检测及人机共存评测。

实施时应确保仿真传感器遮挡和点云噪声贴近真实，并关注目标分割初始化失败。扩展到功能性任务还需增加目标选择、抓取后检查和恢复行为。

#### 总结

核心思想是让全身教师教会单一视觉学生抓取。
1. 训练全场景控制器和局部抓取专家。
2. 在仿真场景中组合教师并生成示范。
3. 蒸馏视觉学生并实机测试。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.28948v1)
- [arXiv](https://arxiv.org/abs/2610.28948v1)

---

<a id='2610.29264v1'></a>
## [Training in Deployment: Human-in-the-loop Online Post-Training for Robot Manipulation](https://arxiv.org/abs/2610.29264v1)

**Authors:** Chun Gu, Xiaofei Wei, Yiming Chen, Bozhou Zhang, Juntong Zhang, Xiatian Zhu, Li Zhang

**Published:** 2026-10-21

**Categories:** cs.RO

**Abstract:**

Large pretrained robot foundation models provide strong priors for manipulation, yet adapting them to target tasks with offline teleoperated demonstrations often fails to cover the contact-rich, recovery, and precision-critical states encountered during deployment. Existing methods mitigate this issue by augmenting offline demonstrations with human-in-the-loop data or online reinforcement learning. Despite this, offline post-training adapts slowly because it cannot immediately absorb deployment experience, while online RL typically relies on value functions or auxiliary actor-critic models, increasing both training complexity and instability. We present TIDE (Training In DEployment), a human-in-the-loop online post-training system. TIDE introduces a provenance-conditioned supervised learning framework that updates the policy online from human interventions and policy-generated rollouts collected during deployment, without a learned critic, reward model, or action scorer. Across four real-world bimanual manipulation tasks, TIDE improves milestone-normalized task progress and binary full-task completion over the warm-start policy and matched additional offline demonstrations, and performs comparably to the fixed-policy collection baseline while making updated checkpoints available during deployment. Each task-specific online post-training session, from the start of deployment rollouts until the final checkpoint is available, completes in less than one hour while learning and checkpoint handling run asynchronously.

### 论文解读

#### 摘要翻译

TIDE是一种部署中人机协同的机器人后训练系统。机器人先运行已有的视觉动作策略，遇到困难时由人接管；系统把两种来源的数据区分后继续更新策略，并在部署会话中逐步交付新模型。

#### 方法动机分析

传统机器人策略通常在部署前完成训练，遇到新状态后要停止收集数据、离线更新，再重新部署。人工纠正虽能提供有效动作，但若不区分纠正与策略自身行为，模型可能混淆两者。作者提出按动作来源记录数据，让策略在真实任务执行期间同步学习，并在安全的episode边界切换模型。

论文把来源标签作为条件变量而非奖励：人类片段提供纠正，策略片段体现当前分布，两类数据都能参与训练但语义不混淆。该区分是在线更新成立的关键假设。

#### 方法设计详解

TIDE以一个3B参数的world-action模型为基础，视频骨干从观察和指令构造世界状态，动作专家输出50步、14维双臂关节相对动作块。机器人客户端执行动作并记录是否由人接管；服务器推理并保存片段，学习器按来源标记进行训练，再将新checkpoint交回客户端。所有可训练模块共同更新，而文本编码器和视频VAE保持冻结。只有经确认成功的episode会提交，所以数据选择会影响学习分布。

具体流程中视频骨干读取图像和指令生成潜在表征，动作专家据此产生关节相对动作块；客户端执行、人工覆盖并记录来源，服务器缓冲数据，学习器联合更新并只在回合结束后切换checkpoint。每次在线优化约0.4秒，训练与执行并行。

#### 方法对比分析

与仅靠人工示范、固定策略或分阶段离线重训相比，TIDE将机器人执行、人工接管、在线学习和模型更新放在同一个部署循环中。其来源标签回答“是谁执行了动作”，不是“动作好不好”；系统仍依赖人类监督，也没有声称每次在线更新都优于当前策略。

#### 实验分析（精简版）

评估包含四项真实双臂任务：插入、叠毛巾、挂毛巾和装链条。每个方法每项任务进行20次无接管测试。TIDE的最终表现与匹配数据量和优化预算的Fixed Policy接近，部分任务更高，关键差别是新模型会在部署会话期间提供。Bit Insertion任务里人类控制占38,839个控制帧的3.69%，这衡量干预负担，不能解释为自主成功率。

作者用20次无接管rollout衡量部署后策略本身，再用干预率描述监督负担；两者不可互换。相近的最终成功表现与更频繁的中途模型交付，体现其流程优势而非绝对更高终点指标。

#### 实用指南

复现需要在线记录动作来源、成功episode筛选、数据缓冲和episode边界checkpoint切换；论文使用50步动作块和单一双臂平台。迁移到其他机器人时，要重新定义动作空间和接管接口，并监测模型更新导致的策略漂移。作者尚未验证跨平台、长期保留或多次会话稳定性。

部署前需要准备任务示范以热启动，并明确成功episode筛选规则、接管按钮和模型回滚方式。跨会话的数据留存与更新稳定性尚未有证据。

#### 总结

核心思想是把人工接管变成部署中的学习信号。
1. 执行并标记人类或策略动作。
2. 在线更新视频与动作模块。
3. 在回合边界换入新策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.29264v1)
- [arXiv](https://arxiv.org/abs/2610.29264v1)

---

<a id='2610.29117v1'></a>
## [Learning to Absorb Corrections: Critic-Weighted Residual Flow for Vision-Based Insertion](https://arxiv.org/abs/2610.29117v1)

**Authors:** Yiqing Yuan, Zhi Li, Jinrong Yang, Mingshan Sun, Mei Yu Zhi, Yong Zhao, Liangliang Ren, Hui Cheng

**Published:** 2026-10-21

**Categories:** cs.RO

**Abstract:**

Vision-based insertion requires correcting alignment errors while retaining useful behavior learned from demonstrations. We present ACT--Flow, an online learning framework that couples an adaptive Action Chunking Transformer (ACT) with a conditional residual flow policy. Following anchor-guided warmup, ACT absorbs successful executed actions through behavioral cloning, while residual targets are recomputed relative to the updated prior. A chunk-level critic learns from successful and failed interactions and evaluates final composed actions. Its estimates weight generated residual candidates for flow matching, complemented by supervision from successful real executions. Candidate scoring is confined to training; deployment uses RGB observations and samples one residual per action chunk. We evaluate the framework on power-plug, USB, and key insertion, together with cable routing and needle threading. Results show a five-task mean success rate of 66% at 20,000 post-initialization interaction steps, exceeding the strongest baseline by 18 percentage points at this checkpoint. Evaluations at matched checkpoints with the residual disabled show ACT's mean success increasing from 16% to 76%. Component ablations further support the complementary roles of behavior absorption, value weighting, and successful-action supervision.

### 论文解读

#### 摘要翻译

ACT-Flow提出一种把示范策略和在线纠正结合起来的机器人学习方法，面向视觉插入、穿线等需要精确接触的任务。它让预训练动作策略提供稳定先验，再由残差策略吸收人类或在线成功纠正，并把改进逐步写回先验。

#### 方法动机分析

只靠模仿学习容易在偏差状态下失败，只靠在线强化学习又可能忘记已有技能或探索成本过高。常见残差方法让修正模块长期独立于基础策略，基础策略难以吸收修正结果。作者的假设是同时更新先验与残差目标，可以在保留示范能力的同时加快纠正学习。

其关键假设是在线成功动作比失败动作更适合作为先验监督，而失败和成功候选仍能帮助critic学习相对价值；因此成功识别质量会直接影响学习方向。

#### 方法设计详解

方法先用锚点示范训练ACT动作块策略，再以成功纠正执行预热残差flow。实际动作是基础动作与残差相加并裁剪。在线成功执行会回流更新ACT，同时依据更新后的先验重新标记残差目标，减少监督随先验漂移而失效。critic从成功与失败缓冲评估多个残差候选，通过softmax权重训练flow matching；成功执行还提供额外监督。每块动作包含4步，flow使用10步积分、每次采样8个候选。

先由锚点示范训练ACT，残差flow对当前先验的动作差进行学习；机器人执行候选动作块后，critic按观测和候选残差估值并形成softmax权重。成功轨迹更新先验，再重新计算残差目标。默认每次生成8个候选、flow积分10步、动作块长度4。

训练流程从锚点示范开始，随后进行在线纠正和先验更新；总评估最高覆盖120k训练步，每个动作块长度4，残差采样使用8个候选。

#### 方法对比分析

与仅做行为克隆相比，该方法可从在线纠正中改进行为；与单独训练残差相比，它把已验证的成功修正吸收到动作先验。它依赖可用的示范锚点、任务阶段结构和成功/失败判定，适合渐进式接触操作，不是无需任务设计的通用强化学习算法。

#### 实验分析（精简版）

五项真实操作任务包括插入、走线和针穿线。3个随机种子下，20k在线步的平均成功率为66%，FTC为48%，提升18个百分点；60k达到87%，120k最终为88%。针穿线仍是较难任务。消融显示禁用残差的ACT部分从16%升至76%，说明先验确实吸收了纠正，但完整方法仍保留残差收益。

评估使用五个真实操作任务、三个随机种子，结果在多个在线步数预算下报告均值；比较同时包括FTC及消融变体，区分最终成功率与学习早期速度。

#### 实用指南

复现要记录哪些动作块真正执行成功、锚点示范如何覆盖阶段，以及先验更新后残差标签如何重算。论文使用固定块长和多候选critic，评估每任务每种子100次。迁移至新操作时应先设计可靠的成功判定和纠正热启动，特别注意失败数据是否代表有用的局部修正。

复现时需保存纠正轨迹、任务阶段和先验版本，避免用更新后的ACT错误解释旧残差。迁移到稀疏成功任务需先设计可靠critic与候选执行安全约束。

#### 总结

核心思想是让先验学会吸收成功纠正。
1. 用示范初始化动作先验。
2. 学习残差并用critic筛候选。
3. 成功动作同时更新先验和残差。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.29117v1)
- [arXiv](https://arxiv.org/abs/2610.29117v1)

---

<a id='2610.29452v1'></a>
## [Distance Field Driven Reinforcement Learning for Reactive and Robust Collision Avoidance](https://arxiv.org/abs/2610.29452v1)

**Authors:** Taiga Ikeshima, Sheila Sutjipto, Lan Wu, Francisco J. Naranjo-Campos, Taiki Ishigaki, Taisuke Kobayashi, Teresa A. Vidal-Calleja, Eiichi Yoshida

**Published:** 2026-10-21

**Categories:** cs.RO

**Abstract:**

Robust and reactive collision avoidance remains a central challenge for deploying Reinforcement Learning (RL) policies on real robotic manipulators, particularly when transferring behaviours learned in simulation to the physical world. This paper presents an RL framework that integrates the Euclidean Distance Field (EDF) directly into the observation space, providing the agent with continuous geometric cues that support stable and responsive avoidance behaviours. By leveraging the rich spatial information encoded in the EDF, the method enables a streamlined reward formulation that achieves effective obstacle avoidance both with and without Artificial Potential Field guidance. The proposed approach is validated on a 7-DoF redundant manipulator, demonstrating zero-shot transfer from simulation to real hardware. Despite being trained exclusively with randomised static objects, the deployed policy generalizes to previously unseen real environments and exhibits robust online avoidance of dynamic obstacles. These results highlight the potential of EDF-based representations to mitigate the sim-to-real gap and support reliable and real-time collision-aware behaviour in complex and changing environments.

### 论文解读

#### 摘要翻译

本文把欧氏距离场（EDF）及其梯度作为强化学习机械臂避障的环境表示，训练策略在接近目标时主动绕开固定或移动障碍。重点是让策略获得连续的空间间隙和避让方向，而不只依赖稀疏碰撞奖励。

#### 方法动机分析

仅靠碰撞发生后的惩罚，策略很难知道应从哪个方向绕开障碍；只给末端距离也无法描述手臂中段与障碍的碰撞风险。作者假设查询机器人多个身体点的距离场，并用势场方向作辅助奖励，可以给PPO更及时且具有几何意义的避障线索。

方法假设可由当前RGB-D观测构造足够准确的距离场，并以少量查询点近似整个机械臂的碰撞风险；相机遮挡会直接破坏此前提。

#### 方法设计详解

系统从RGB-D观测重建环境距离场，或在仿真直接使用几何信息。它在末端执行器以及另外四个代表性身体位置查询最近障碍距离和距离梯度。PPO将这些量与机器人状态、目标位置共同作为观测，输出Franka FR3七个关节的增量动作。APF变体增加方向对齐奖励；训练在含随机球形障碍的MuJoCo环境中进行。真实部署不需要再微调策略。

距离场为每个查询点返回最近表面间距及其梯度方向，网络将这些特征与关节、目标信息融合，PPO输出小幅关节角变化。APF奖励再依据目标方向和势场方向的一致性提供更密集的引导。

#### 方法对比分析

与仅在末端取样的方案相比，全身采样提供中间连杆周围的局部安全线索；APF奖励则把避障方向直接纳入学习目标。这两种设计分别作用于观测和奖励，能够组合使用。适用范围是机械臂无碰撞到达目标，论文没有评估抓取、复杂接触或人机协作的安全要求。

#### 实验分析（精简版）

在固定障碍真实实验中，每种配置10次：仅末端EDF成功率60%，末端加APF为40%，全身EDF为70%，全身EDF加APF也为70%。这些结果显示全身距离信息有帮助，但APF奖励并未在实机每种比较中提升成功率。作者还展示动态障碍交互；单个前向RGB-D相机的遮挡会遗漏障碍后的机械臂碰撞。

仿真训练含随机球形障碍，并在真实静态障碍与动态障碍场景检查迁移。固定障碍指标按无碰撞到达计算，不将仅完成接近或过程中的碰撞算作成功。

#### 实用指南

复现需统一距离场建模、查询点位置、PPO训练随机化和碰撞终止条件；实机实验使用FR3与单目RGB-D观测。向移动操作迁移时，需要处理自遮挡、动态深度缺失和更密集的身体采样，并单独验证实体碰撞率，不能把零样本策略视为安全认证。

实践中要选取能覆盖整臂的查询点，处理深度相机盲区并检查距离场实时更新开销；面向抓取还需考虑末端接触目标与障碍物的区别。

#### 总结

核心思想是用距离梯度教会机械臂避障。
1. 从场景构造距离场。
2. 查询末端和身体点的间距方向。
3. 以PPO学习到达并避碰。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.29452v1)
- [arXiv](https://arxiv.org/abs/2610.29452v1)

---

<a id='2610.29438v1'></a>
## [A2I-Calib++: A Kinematics-Driven Anti-noise Active Calibration Framework for Whole-Body Multi-IMU Systems on Legged Robots](https://arxiv.org/abs/2610.29438v1)

**Authors:** Fangyu Jiang, Chaoran Xiong, Kehui Ma, Zhen Sun, Zeyu Zhang, Ling Pei

**Published:** 2026-10-21

**Categories:** cs.RO

**Abstract:**

Recently, multi-node inertial measurement unit (multi-IMU) systems have shown promise for robust odometry in legged robots. Multi-IMU fusion can improve legged robot odometry but requires accurate spatiotemporal calibration among the sensors. Existing multi-IMU calibration methods, designed for aerial or handheld platforms, face challenges on legged robots because routine locomotion may provide insufficient excitation, while foot slips, impacts, and vibrations introduce measurement disturbances. To address these challenges, we propose A2I-Calib++, a kinematics-driven anti-noise active calibration framework for body-mounted IMUs covered by the robot kinematic model. It estimates temporal offsets, extrinsic rotations and translations, and inertial biases through temporal correlation search followed by analytic spatial and bias estimation, without artificial calibration targets or complex parameter tuning for the target IMU. A2I-Calib++ comprises three core modules: 1) a Noise-Isotropy-Inspired Trajectory supported by a finite-horizon conditioning proposition that generates sufficient excitation to limit noise sensitivity; 2) a constraint-aware Adversarial Motion Priors controller that combines the anti-noise excitation with a rolling-contact reward for contact-aware execution; and 3) a contact-constrained virtual IMU that fuses pre-calibrated calf IMUs and joint kinematics under a rolling-contact model to construct the calibration reference. Furthermore, evaluations on both simulation and real-world quadruped platforms demonstrate notably reduced noise sensitivity and simulation calibration errors compared to conventional methods, together with repeatable real-world calibration and improved proprioceptive odometry accuracy for legged robots.

### 论文解读

#### 摘要翻译

A2I-Calib++是一套面向四足机器人机身与腿部多IMU系统的主动时空标定方法。它让机器人执行专门设计的抗噪运动，再利用关节运动学和足端接触信息构造参考，从而估计不同IMU之间的时间偏移、旋转、平移及偏置。

#### 方法动机分析

常规标定轨迹可能缺少足够激励，使旋转和平移参数难以区分；腿足滑动或接触误差也会污染参考运动。作者提出通过噪声各向同性轨迹改善问题条件数，并用接触约束的运动策略执行该轨迹，再结合腿部运动学给IMU标定提供额外观测。

其核心假设是关节编码器与足端接触可构成比外部参考更稳定的机身运动约束；若地面持续滑移，虚拟IMU参考和标定精度都会受影响。

#### 方法设计详解

首先，NIT根据旋转和平移残差设计抗噪参考轨迹。随后约束感知AMP在足端接触、防滑等条件下让四足机器人执行；由腿关节和机身状态通过前向运动学构造多个腿部虚拟IMU（VIMU），再利用纯滚动接触模型融合为参考。估计阶段先搜索时间偏移，再以SVD求外参旋转和陀螺偏置，最后用最小二乘恢复平移和加速度计偏置。

流程先优化激励轨迹的噪声条件数，随后由AMP使机器人在接触条件下执行；多个腿部虚拟IMU经过纯滚动约束构成参考，时间相关搜索完成后，SVD求旋转、最小二乘求平移与偏置。

#### 方法对比分析

与依赖人工手持激励或单一刚体运动的标定相比，A2I-Calib++利用四足机器人自身运动学和多腿接触约束构造参考，并通过优化轨迹降低噪声敏感性。方法适用于具有可靠关节状态和足端接触信息的腿式平台；轮足打滑、软地面和不同形态平台需要重新验证其接触模型。

#### 实验分析（精简版）

仿真中时间偏移RMSE为0.19毫秒，外参旋转误差0.177度、平移误差0.287厘米。真实Go2评估以FAST-LIO2轨迹作参考，下游里程计APE为0.366米、RRE为3.277，略好于同设置iKalibr的0.370米和4.347。真实外参没有绝对地面真值，因此后者是下游代理验证，不能视作绝对标定误差。

仿真提供已知真值下的参数误差，真实平台则用重复采集和下游里程计评估。因为真实外参没有绝对测量基准，论文谨慎使用FAST-LIO2作为参考轨迹，而不把它当作完美真值。

仿真时间偏移误差为0.19 ms，旋转误差0.177度；真实结果是下游轨迹比较，不能当作绝对外参真值。

#### 实用指南

复现需准确同步IMU、关节和接触数据，执行约3.2秒的标定运动，并检查地面摩擦是否支持纯滚动假设。论文在Go2平台上进行了真实评估。迁移到其他四足机器人时需要重建腿部运动学和接触约束，并关注不同地面造成的参考偏差。

实际使用需同步时间戳、检查接触状态估计并确认地面条件；平台改变腿长、IMU位置或采样率后都要重新计算运动学和标定激励。

#### 总结

核心思想是用抗噪运动和足端约束标定多IMU。
1. 生成条件良好的激励轨迹。
2. 由关节和接触构建虚拟IMU参考。
3. 搜索时间并求解外参与偏置。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.29438v1)
- [arXiv](https://arxiv.org/abs/2610.29438v1)

---

