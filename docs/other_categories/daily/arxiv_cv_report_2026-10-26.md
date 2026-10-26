time: 20261026

# Arxiv Computer Vision Papers - 2026-10-26

## Executive Summary

## 今日研究脉络

这批十篇论文中，机器人与具身学习占据主线：研究者尝试让模型更有效地利用仿真、结构化动作先验、视觉空间线索和交互数据。DiscoDemo 将特权状态、演示动作与仿真重置结合，降低真实机器人采集示范的成本；DynaSlots 用时间一致的软槽位学习物体交互和潜在动作，再迁移到策略学习。Horizon 论文进一步提醒，开放环规划长度与每步采样深度存在耦合，单独增加搜索预算未必带来更好动作。TravSUITE 将视觉语义、几何和速度分布用于越野路线规划，体现了从室内机械臂向复杂户外环境扩展的方向。

另一组工作关注多模态模型与视频世界建模。Ψ-0.6 将视觉、几何和语言任务纳入统一模型，并允许按任务组合推理路径；Tempo-Zero 则把问题生成绑定到可执行环境和采样帧上的可核验标签，再通过 Questioner 与 Solver 的联合训练逐步扩展课程。UniWorld 面向多来源视频，用滚动深度状态与重叠窗口降低长视频续生成成本。它们共同展示了从静态问答走向可交互、可延续或可验证的时序视觉任务，但评测协议和真实环境覆盖仍是判断泛化能力的关键。

其余论文分别改善多技能操作和视觉解码效率。MAST-VLA 将技能条件化的 LoRA-MoE 与轨迹引导的空间 affordance 先验结合，直接针对多技能共享策略中的干扰；RoboCode 以共享软动作原语压缩示范表示，在仿真任务中提升模仿效果。TileShare 的观察则是：重叠切块带来的冗余主要存在于逐 token 运算，注意力仍应保留原有 tile 上下文；因此可共享归一化、投影和 MLP 等计算，同时保留窗口边界。

值得优先精读的论文取决于读者关注点：机器人落地可先看 DiscoDemo、MAST-VLA 与 TravSUITE，并结合 Horizon 的预算耦合分析审视实验设计；关注可验证视频推理，可读 Tempo-Zero；关注世界模型与长视频生成，可读 UniWorld；做视觉系统优化则优先看 TileShare。整体趋势是把模型能力与数据生成、执行约束、时序状态或硬件计算结构共同设计。各论文的结果来自不同数据集、仿真器和机器人协议，数值不宜横向直接比较；尤其要区分仿真成绩、有限实机试验与纯离线视频生成评测。

---

## Table of Contents

1. [DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning](#2610.32093v1)
2. [Coupling Execution Horizon and Sampling Depth in Flow-Matching VLA Policies](#2610.32102v1)
3. [TravSUITE: Traversability via Self-Supervised, Uncertainty-Aware IRL and Terrain Estimation](#2610.32130v1)
4. [RoboCode: Learning Reusable Action Primitives for Robot Manipulation](#2610.32225v1)
5. [MAST-VLA: Skill-Conditioned Adaptation with Trajectory-Affordance Priors for Multi-Skill Robot Manipulation](#2610.32312v1)
6. [DynaSlots: Slot-Centric 3D Dynamics Pretraining for Robot Manipulation](#2610.32330v1)
7. [$Ψ$-0.6: Adaptive Streams of Thought in a Unified Vision-Language World Model](#2610.32177v1)
8. [UniWorld: A Unified World Model for Real-Time Visual World Simulation](#2610.32381v1)
9. [TileShare: Training-Free Acceleration of Overlapping Tiled ViT Decoding](#2610.32233v1)
10. [Tempo-Zero: Recursive Self-Play in Executable Worlds for Video Reasoning](#2610.32433v1)

---

## Papers

<a id='2610.32093v1'></a>
## [DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning](https://arxiv.org/abs/2610.32093v1)

**Authors:** Minho Park, Kinam Kim, Donghu Kim, Byungkun Lee, Dongyoon Hwang, Yongjae Shin, Junha Hyung, Hojoon Lee, Jaegul Choo

**Published:** 2026-10-23

**Categories:** cs.RO

**Abstract:**

Behavior cloning for robotic manipulation is highly sensitive to the coverage of its training data, yet collecting diverse successful demonstrations remains costly. We study reinforcement learning (RL) not as a deployable controller, but as a data generator for downstream visuomotor imitation learning. Our key premise is that multiple successful solution modes cover a broader range of successful trajectories than repeatedly collecting a single narrow solution. We introduce DiscoDemo, which trains a privileged latent-conditioned RL generator with a parallel reverse curriculum and a success-gated diversity objective to discover diverse solutions while preserving task success. The trained generator produces multiple successful closed-loop behaviors, which are safety-filtered and exported as a static dataset for downstream imitation learning. Across four manipulation tasks, DiscoDemo produces broader successful-state coverage and improves downstream visuomotor policies in both simulation and the real world. Averaged across tasks, downstream success increases to 62% from 40% for the strongest baseline in simulation, and to 78% from 39% on the real robot, respectively. These results suggest that successful solution diversity improves imitation learning for a pretrained flow-matching VLA and remains complementary to corrective data collection. Project page: https://davian-robotics.github.io/DiscoDemo

### 论文解读

#### 摘要翻译

本文提出 DiscoDemo：在仿真中训练一个能从特权状态生成成功操作轨迹的策略，再把轨迹筛选成视觉模仿学习数据。作者希望降低逐任务采集示范的成本，同时扩展成功轨迹的行为覆盖。

#### 方法动机分析

普通强化学习追求单条策略成功，生成数据时却可能只得到少数相似解；无约束的多样性又会引入失败动作。论文的核心假设是：先用成功奖励守住任务正确性，再只鼓励成功轨迹之间的差异，就能扩大演示覆盖而不牺牲可用性。

#### 方法设计详解

流程分三段。先让一个特权生成器读取仿真状态和潜变量，使用并行反向课程逐步覆盖更难的初始状态；成功奖励判断任务是否完成，成功门控的 METRA 式目标鼓励不同潜变量形成不同状态访问轨迹，安全项惩罚接触和关节异常。训练后冻结生成器，采样大量轨迹，并用任务安全过滤器去掉不合格示范。最后只把筛选出的视觉与动作序列用于训练 π0.5 行为克隆策略。论文在四个 Franka 桌面任务上，以每任务 20 条人类示范作比较；生成阶段并行运行 2,048 个环境。

#### 方法对比分析

与单纯扩大采样或直接奖励状态差异相比，DiscoDemo把成功门控放在多样性奖励前面，并把“生成器可访问的仿真状态”与“部署学生能见到的观测”分开。它适合有可重置仿真器、明确成功信号且能构造安全过滤器的操作任务；对不可模拟接触或奖励难定义的任务，数据质量保证会变弱。

#### 实验分析（精简版）

作者报告仿真平均成功率 62%，高于最佳基线 40%；用生成数据训练后，真实机械臂平均成功率为 78%，最佳基线为 39%。真实测试为四个任务各 20 次，样本仍小，且仅覆盖桌面操作。结果支持数据生成有价值，但不证明对长时序、多物体或不同机械臂平台都能迁移。

#### 实用指南

复现需实现并行仿真、特权策略、安全代价和任务级过滤，生成阶段报告采用 5,000 万环境步；学生在 π0.5 上微调 20,000 步，三随机种子。应先确认仿真接触与现实一致，再逐项检查筛选器是否会删除关键成功模式。论文表示将公开实现，但不能据此假定代码现已可用。复现实验还应报告碰撞与关节超限等安全指标，避免只按任务成功率筛选而漏掉风险。

#### 总结

核心思想：成功门控的多样化演示。
1. 仿真器提供状态与成功奖励。
2. 潜变量策略在课程中探索多种成功轨迹。
3. 安全过滤并导出视觉动作示范。
4. 用筛选数据训练视觉策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32093v1)
- [arXiv](https://arxiv.org/abs/2610.32093v1)

---

<a id='2610.32102v1'></a>
## [Coupling Execution Horizon and Sampling Depth in Flow-Matching VLA Policies](https://arxiv.org/abs/2610.32102v1)

**Authors:** Jiabing Yang, Yuan Xu, Yixiang Chen, Qisen Ma, Peiyan Li, Kai Wang, Chenhang Cui, Tao Yu, Zichen Wen, Wen Ye, Bowen Fang, Ziheng He, Xiangnan Wu, Yingda Li, Yan Huang, Liang Wang

**Published:** 2026-10-23

**Categories:** cs.RO

**Abstract:**

Flow-matching Vision-Language-Action (VLA) models generate expressive continuous trajectories for robotic manipulation, yet balancing real-time responsiveness with execution stability remains a persistent challenge. Standard deployment workflows typically treat numerical solver depth and execution horizon as decoupled choices. The horizon amortizes inference latency, while solver depth is kept high to preserve integration fidelity. In this paper, we reveal a fundamental compute-execution coupling in flow-matching VLAs governed by a systematic sign reversal. While coarse numerical integration degrades precision under rapid replanning, it markedly improves task performance under extended open-loop commitment. Geometrically, numerical truncation contracts lateral action dispersion around an invariant trajectory centroid, serving as an implicit regularizer that suppresses compounding open-loop drift. Across LIBERO, RoboTwin 2.0, and real-world Franka manipulation on representative VLA backbones including $π_0$, $π_{0.5}$, and GR00T, exploiting this joint coupling consistently achieves superior task success while drastically lowering inference latency, entirely training-free and without architectural modifications. These findings establish numerical sampling depth as an operational variance regulator, shifting its role from a static precision budget to an active deployment knob in robot learning. Project page: https://jiabingyang01.github.io/Horizon-Depth-VLA/

### 论文解读

#### 摘要翻译

本文研究流匹配视觉语言动作策略的数值采样步数，如何与动作执行时域共同影响操纵成功率。作者发现“少步更快但更差”并非普遍规律：效果随机器人多久重新观测并规划一次而翻转。

#### 方法动机分析

连续动作的流匹配策略通常通过多次 ODE 求解得到动作序列，但计算成本高。核心动机是减少每个动作块的求解步数，同时保证轨迹稳定。作者假设粗采样会收缩动作分布的方差；这种收缩在长时间开环执行时能抑制漂移，在频繁闭环反馈时却会损害绕障所需的多样性。

#### 方法设计详解

其流程包括锁定策略、配对扫描两个控制轴、记录成功率与动作云变化，再通过定向干预验证机制。

方法先固定已训练好的 VLA，不改权重，对执行时域和 Euler 采样数做因子实验。对同一策略、任务和随机种子，分别改变“每次生成后执行多少步”与“流场积分几步”，记录成功率和动作云几何。再用噪声重标定和跨重规划固定噪声等干预，区分一般方差缩小与沿学习到的流场收缩。验证覆盖 π0、π0.5、GR00T-N1.7，LIBERO 的四类任务及 RoboTwin，另外做了实体机器人实验。每个条件的配对种子有助于把采样效应与任务随机性区分。训练/推理报告包含 8 个 H100 GPU 的实体实验资源设定及不同采样配置的逐次延迟。

#### 方法对比分析

与仅报告推理延迟或固定采样次数的工作不同，这篇论文把控制反馈频率作为关键实验变量，并用成对干预支持机制解释。结论适用于生成式动作策略：采样预算应根据开环承诺长度选，而不能只按模型或任务固定。

#### 实验分析（精简版）

实验设定覆盖55个模拟任务，并在八张 GPU 上训练相关模型；推理部分则直接改变固定策略的 Euler 求解步数。

在 LIBERO-Long 的 π0 上，执行 20 步时，2 次 Euler 采样成功率为 90.5%，10 步为 86.9%；执行 5 步的短闭环条件下，2 步采样为 78.5%，反而低于 10 步的 81.2%。跨三个模型的效应方向相似，最多报告 5.6 倍速度提升。实体实验覆盖多个操作环境，但任务规模与动态扰动类型有限。

#### 实用指南

实际部署可先测量反馈周期与动作块长度，再分别校准采样深度；对长开环段优先验证少步稳定性，对高频重规划任务保留足够采样精度。论文报告 55 个模拟任务和 2400 次实体试验，代码与硬件依赖需按原协议复现。该策略无需再训练，但机械臂控制频率、求解器实现和观测时延都会改变适用边界。

#### 总结

核心思想：采样步数随执行时域选。
1. 固定 VLA，改变采样步数。
2. 同时扫描动作执行长度与重规划频率。
3. 用干预检验动作方差收缩机制。
4. 按闭环需求配置速度与稳定性。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32102v1)
- [arXiv](https://arxiv.org/abs/2610.32102v1)

---

<a id='2610.32130v1'></a>
## [TravSUITE: Traversability via Self-Supervised, Uncertainty-Aware IRL and Terrain Estimation](https://arxiv.org/abs/2610.32130v1)

**Authors:** Samuel Triest, Amirreza Shaban, David D. Fan, Wenshan Wang, Sebastian Scherer

**Published:** 2026-10-23

**Categories:** cs.RO

**Abstract:**

Traversability analysis in off-road settings remains a fundamental challenge for mobile robots. Key difficulties include constructing an accurate and expressive local map from multimodal sensor data, and using the map to design traversability rules that yield desirable navigation behavior. Importantly, this system must be resilient to the limited sensing regime brought about by complex environments and high speeds. In this paper, we present TravSUITE, a traversability system suitable for high-speed navigation in off-road environments. TravSUITE consists of two major components: 1) an efficient voxel mapper that leverages visual foundation models (VFMs) to build a rich geometric-semantic local map from streams of on-board sensor data, and 2) a unified neural network that jointly predicts traversability-relevant quantities in bird's eye view (BEV), including geometry, semantics, speed and cost. Our training strategy is entirely annotation-free and self-supervised, leveraging tasks such as map inpainting and inverse reinforcement learning (IRL) to learn both map representations and traversability. We perform a thorough ablation study and comparison to state-of-the-art approaches, and the results indicate that cost learning and auxiliary inpainting each contribute significantly to planning quality, and their combination is critical for achieving state-of-the-art performance in off-road path planning. We also design a simple risk adaptation mechanism to leverage our method's uncertainty estimates at deploy-time, and demonstrate that a combination of inpainting and risk estimation can result in 80% fewer navigation errors and 5% faster autonomous traversal speeds in real-world hardware experiments.

### 论文解读

#### 摘要翻译

TravSUITE研究如何从无标注越野驾驶数据构建可用于规划的可通行性表示。系统在车体坐标的三维体素地图上预测地形几何、视觉基础模型语义和速度/代价分布，再用风险敏感规划选择路线。

#### 方法动机分析

传统可通行性图往往依赖人工代价标签，单一几何或视觉特征又难同时描述泥地、植被与障碍。作者的动机是利用车辆自身行驶轨迹提供弱监督，并将预测不确定性纳入规划；核心假设是专家选择的行驶路径及其绕行可反推出地形代价。

#### 方法设计详解

这条 pipeline 的输出并不是一个二值可通行标签，而是多个与规划相关的连续预测头；CVaR 在候选轨迹层面读取代价分布的尾部，因此可用同一地图比较平均速度与潜在高风险。

首先用激光雷达惯导里程计估计状态，并把相机、几何和语义特征融合到局部体素地图。网络包含几何、VFM特征、语义和速度多头。训练时对已观测地图区域做 hindsight inpainting，使网络从部分地图预测完整成本图；再以最大熵逆强化学习从专家轨迹反推代价。在线规划器用 CVaR 处理速度和风险分布，在候选轨迹上权衡行驶时间与碰撞风险。数据约十万帧、两处场地，约四小时采集，训练五轮。

#### 方法对比分析

与仅拟合几何或语义分类的路线不同，TravSUITE把多模态地图补全、轨迹偏好学习和尾部风险约束放在同一链路中。它适用于局部地图能够持续更新、规划器可评估备选轨迹的越野平台；目前不是通用全局建图系统。

#### 实验分析（精简版）

复现实验还应区分测试地点、天气和障碍类型，并报告人工干预，因为少碰撞可能来自更保守的低速策略。

离线测试显示地图补全提升预测质量；实体 ATV 在约 1.4 公里路线完成行驶。并行遍历加 CVaR 报告零碰撞、一次人工干预，平均 4.49 米/秒。证据依赖两个地理区域和有限季节条件，不能据单次路线证明不同底盘、天气或地形上的安全保证。

#### 实用指南

复现需要同步标定激光雷达、惯导与相机，建立时间一致的局部体素地图，并保留专家轨迹和干预事件。作者使用 10Hz 地图、五轮训练、batch size 3。移植到自动驾驶可考虑以道路边缘、接管和舒适度轨迹反推代价，但城市道路传感器遮挡、交通参与者交互和法规约束尚未由本研究验证。

#### 总结

核心思想：从行驶轨迹学习风险地图。
1. 感知构建多模态局部体素。
2. 补全地图并从专家轨迹反推代价。
3. 用 CVaR 规划速度与安全折衷。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32130v1)
- [arXiv](https://arxiv.org/abs/2610.32130v1)

---

<a id='2610.32225v1'></a>
## [RoboCode: Learning Reusable Action Primitives for Robot Manipulation](https://arxiv.org/abs/2610.32225v1)

**Authors:** Xianyue Peng, Devi Amarsaikhan, Iman Soltani

**Published:** 2026-10-23

**Categories:** cs.RO, cs.LG

**Abstract:**

Recent robot manipulation policies commonly predict continuous action chunks directly from observations, without explicitly modeling reusable structure in the action space. However, manipulation demonstrations often contain recurring short-horizon motion patterns that are shared across tasks. We introduce \textsc{RoboCode}, which learns these patterns through a shared soft-gated primitive action interface. Rather than assigning each action chunk to a single discrete code, \textsc{RoboCode} represents it as a soft composition of multiple learned primitives, providing a flexible intermediate representation of short-horizon actions. The primitive interface is learned from action chunks pooled across multiple tasks and then frozen while task-specific state encoders are trained to map current state inputs into the shared latent space.   Experiments on five ManiSkill manipulation tasks address three questions. First, the learned primitive interface reconstructs action chunks with an average reconstruction MSE of 0.0187. Second, analysis of code activations reveals structured code reuse across tasks. Third, \textsc{RoboCode} achieves the highest or tied closed-loop success rate on all evaluated tasks, with an average success rate of 75.6%, compared with 68.8% for ACT and 41.6% for the larger Diffusion Policy baseline. These results suggest that soft-gated primitive codebooks provide an effective and interpretable intermediate action representation for robot manipulation.

### 论文解读

#### 摘要翻译

RoboCode提出可复用的机器人动作原语接口：先从多任务示范中学习共享动作码本，再冻结码本，让每个任务把观测状态映射为原语组合。目标是在多任务行为克隆中减少动作冗余，并让不同任务共享可解释的运动片段。

#### 方法动机分析

常见策略直接从观测回归长动作块，任务间容易相互干扰；离散硬量化又可能把连续动作切得过粗。作者假设多个操作任务可以由一组共享原语组合表达，而低层动作接口固定后，任务专属状态编码器更容易学习何时调用哪些原语。

#### 方法设计详解

输入包括归一化后的机器人状态、物体位姿和目标信息；输出仍是连续关节或夹爪动作，而不是离散技能ID。软门控可同时激活多个码字，使相近动作通过组合表示，但也让码本可解释性需要额外分析。

第一阶段把五项任务的动作块汇总，动作编码器将每个长度 16 的片段嵌入 64 维表示，并通过与 32 个原语码本的余弦相似度形成软门控；共享解码器将原语加权组合还原连续动作。训练中包含重建、投影、稀疏和多样性目标。第二阶段冻结码本和动作解码器，每项任务独立训练状态编码器，把物体状态、本体状态转成原语门控。推理采用闭环滚动规划，每执行 8 步就重新观测和预测。

#### 方法对比分析

与 ACT 或扩散策略直接预测动作序列不同，RoboCode显式分离“动作空间结构”与“任务状态到动作结构的映射”。它适用于演示任务中存在重复运动模式、且可以取得物体状态的操作场景；码本质量不足会限制后续所有任务。

#### 实验分析（精简版）

优势是共享接口不随任务变化，任务专属编码器负责观测差异；代价是第一阶段数据必须覆盖后续任务所需动作，未见动作不容易靠第二阶段补出来。

五项 ManiSkill 任务共 4,620 条示范，RoboCode 平均成功率为 75.6%，ACT 为 68.8%，较大的 DP-Large 为 41.6%；动作片段重建 MSE 为 0.0187。实验均在仿真，比较优势不能视为真实机器人验证。

#### 实用指南

第一阶段学到的表示是共享瓶颈，因此在新任务接入前应先验证其动作重构质量与技能覆盖。

复现需先统一动作尺度与动作块长度，再监控码本重建误差和原语使用分布。论文设 H=16、stride=4、32 原语、64 维码本；第一阶段 400 epoch，第二阶段 200 epoch。迁移至跨平台机器人时，需统一坐标系和低层控制接口；若任务动作语义差异过大，强制共享码本可能产生负迁移。

#### 总结

核心思想：共享动作原语，任务专属调用。
1. 多任务动作块训练软码本。
2. 冻结原语接口。
3. 每个任务学习状态到门控。
4. 闭环执行并定期重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32225v1)
- [arXiv](https://arxiv.org/abs/2610.32225v1)

---

<a id='2610.32312v1'></a>
## [MAST-VLA: Skill-Conditioned Adaptation with Trajectory-Affordance Priors for Multi-Skill Robot Manipulation](https://arxiv.org/abs/2610.32312v1)

**Authors:** Jiachi Yang, Zhiyi Xu, Hongquan Zhang, Xin Tan, Zhizhong Zhang, Jingyu Gong, Yuan Xie

**Published:** 2026-10-23

**Categories:** cs.RO

**Abstract:**

Generalist robot manipulation requires a single vision-language-action (VLA) policy to execute diverse skills across scenes and instructions. However, current multi-skill VLA policies often leave two aspects of an instruction insufficiently modeled: which primitive skill to execute and where/how to contact and move objects. We propose MAST-VLA (Multi-skill Adaptation with Skill-conditioned experts and Trajectory-affordance priors), a dual-conditioning framework for robot manipulation. MAST-VLA introduces skill-conditioned LoRA Mixture-of-Experts (LoRA-MoE) adapters in the LLM backbone, where skill-query tokens and routing biases guide low-rank expert specialization, and converts a predicted affordance heatmap into a patch-level trajectory-affordance spatial prior for the action head. On the CALVIN ABC$\rightarrow$D long-horizon multi-skill cross-scene generalization setting, MAST-VLA achieves state-of-the-art performance, reaching an Avg. Len of 4.65 over five-subtask evaluation sequences. Real-robot multi-skill sequence experiments further validate its deployability under real perception and control noise.

### 论文解读

#### 摘要翻译

MAST-VLA面向需要连续完成多种操作技能的机器人任务，提出把语言指令中的原语技能结构与演示轨迹中的空间交互区域同时用于适配视觉语言动作模型。它希望降低技能切换时的干扰，并让动作解码关注未来接触位置。

#### 方法动机分析

单一 LoRA 或任务级专家路由未必区分“旋转、推、抓、放”等动作原语，普通视觉注意也可能只看当前接触点而忽略后续运动。作者假设技能条件能引导专家分工，轨迹推导的 affordance 热图则能为空间动作解码提供局部先验。

#### 方法设计详解

该设计的输入到输出包含两条并行通路：语言侧决定技能查询和专家偏置，几何侧把未来动作轨迹变为空间注意力先验，最后由统一动作头融合。作者用 CALVIN、模拟消融及真实机械臂任务评估。

模型以 Qwen2.5-0.5B 为视觉语言骨干，先把指令匹配到粗粒度原语技能，技能查询向量与路由偏置控制 LoRA-MoE 的专家调用；再从演示的未来末端执行器轨迹和点云生成 affordance 热图，把热图转成注意力偏置注入动作头。模型输出 8 步动作块，执行中持续观察和重规划。训练包含技能条件适配和轨迹空间监督，使用 CALVIN 与实体机械臂数据。

#### 方法对比分析

与只依赖语言条件的 VLA 相比，MAST-VLA为“技能模块选择”和“交互区域选择”分别设计条件路径；不需要把语言专家绑定为单一技能。适用于多技能、桌面操作、可获得轨迹示范和 RGB-D 标定的场景。

#### 实验分析（精简版）

实物结果应结合任务长度与成功定义理解；不同的 Success@3 和 Success@5 不能直接同一量纲比较。

在 CALVIN ABC→D 上，平均长度 4.65，Success@5 为 84.7%，高于所列 VLA-Adapter-Pro 的 80.0%。实体机器人测试 160 次，Success@3 为 70.6%。结果说明模拟到现实有一定迁移，但规模有限；深度遮挡、坐标标定与低层执行偏差仍会影响热图和动作。

#### 实用指南

部署时应重点检查深度外参、点云坐标系、任务技能词表及动作重规划间隔。

实用上可把热图当作解码偏置而非硬目标区域，这样策略仍可结合视觉上下文调整；论文的消融显示两种条件路径都提供作用。

复现需统一语言技能分类、末端轨迹坐标和 RGB-D 点云标定；模型报告 50,000 次微调更新、每卡 batch 16、8 张 GPU，LoRA rank 64。实际部署要保留反馈重规划，因为论文没有显式纠正执行偏差的模块。迁移到移动操作或自动驾驶场景，需要重新定义原语集合与空间 affordance，不宜直接复用桌面技能分类。

#### 总结

核心思想：技能路由叠加轨迹空间先验。
1. 从指令识别粗粒度原语技能。
2. 技能查询调节 LoRA 专家路由。
3. 轨迹生成 affordance 注意偏置。
4. 解码动作块并闭环重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32312v1)
- [arXiv](https://arxiv.org/abs/2610.32312v1)

---

<a id='2610.32330v1'></a>
## [DynaSlots: Slot-Centric 3D Dynamics Pretraining for Robot Manipulation](https://arxiv.org/abs/2610.32330v1)

**Authors:** Meng Li, Lijiang Chen, Qi Zhao, Jun Jiang, Longhao Zou, Shuchang Lyu, Guangliang Cheng

**Published:** 2026-10-23

**Categories:** cs.RO

**Abstract:**

Robot manipulation involves localized changes: picking up a cup moves the cup while leaving the surrounding scene largely unchanged. Yet existing 3D visual pretraining methods, including recent dynamics-aware variants, model state transitions at the scene level, compressing the motion of everything in a scene into a single global latent action. This conflates semantically distinct transitions in scenes with multiple moving parts, obscures dependencies between changing regions, and leaves the learned representation vulnerable to task-irrelevant scene content. We introduce DynaSlots, an action-free, label-free pretraining framework that decomposes each point cloud frame into a small set of temporally consistent slots and models dynamics at this per-slot level rather than the scene level. A temporal slot propagation module promotes consistent slot assignments across frames; an inverse dynamics model then extracts one latent action per slot from feature differences, and a diffusion-based forward dynamics model, paired with a self-supervised interaction head, predicts each slot's future state while learning pairwise proximity-and-motion relations. Experiments on Meta-World, Adroit, and a RealMan RM65-B robot demonstrate that DynaSlots outperforms prior scene-level baselines. It achieves superior success rates in simulation and exhibits strong robustness to scene clutter and unseen objects in real-world manipulation. Code is available at \href{https://zhucezhanghao12345-source.github.io/DynaSlots-Anonymous/}{https://zhucezhanghao12345-source.github.io/DynaSlots-Anonymous/}.

### 论文解读

#### 摘要翻译

DynaSlots提出面向机器人操作的对象中心 3D 动力学表示。模型在点云序列中维护少量可软分配的时序槽位，把场景变化编码为每个槽位的潜在动作，再预测未来槽位状态并迁移到策略学习。

#### 方法动机分析

无监督世界模型常把整幅场景压成单个潜变量，难以分离“物体运动”和“背景不变”；对象分割监督又昂贵。作者的假设是，一组跨时间稳定的软槽位可自发捕获局部物体变化，用槽位特征差分近似潜在动作，再借助槽位间关系学习交互结构。

#### 方法设计详解

输入点云由全点 Transformer 编码，不依赖远点采样或邻域分组；绑定器每帧迭代三轮，让点与上一时刻槽位竞争。下游扩散策略把槽位条件与机器人状态联合输入，预测后续动作序列。

输入为过去、当前和未来点云，经共享 Point Transformer 得到点特征；Temporal Slot Binder 使用竞争分配将点绑定到 K=6 个软槽。逆动力学模块从同一槽位在不同时间的特征差预测潜在动作，跨槽编码器和扩散前向动力学预测当前槽位的未来状态。EMA 目标分支提供自监督目标，交互头根据质心接近与共同运动产生槽位关系伪标签。预训练编码器与绑定器随后给下游扩散策略提供槽位条件。

#### 方法对比分析

与逐帧分割或显式对象追踪相比，该方法不要求对象标签，槽位绑定跨帧维持对应关系；与全局视频潜变量相比，它保留对象级局部变化。适用于点云观测、对象交互明显且场景复杂度可由少量槽位覆盖的操作任务。

#### 实验分析（精简版）

结果支持该表示在所测操作任务有效，但不能证明槽位对应稳定物体身份；任务成绩和分割质量也不是同一个指标。

在 MetaWorld 上成功率 80.8%，比 AFRO 高 4.8 个百分点；Adroit 达 88%，高 5 个百分点。四项实体任务平均 79%，基线 70.5%，但仅有五种初始布局、每布局十次试验。拥挤物体和未见物体也有提升，然而真实环境仍然有限。

#### 实用指南

作者用四个独立随机种子选择策略检查点，并在 MetaWorld 按任务难度报告结果；这些设计让统计较完整，但并未替代跨平台重复。

复现需要点云坐标规范、时序采样和槽位绑定实现。论文用 K=6、槽位维度 64、潜在动作维度 16，预训练 300 epoch，策略训练 100 epoch。注意“接近与共同运动”关系是伪标签，不等价于真实接触。迁移到自动驾驶可设车辆/道路参与者槽位，但远距交互、遮挡和对象出生消失会考验绑定稳定性。

#### 总结

核心思想：用时序槽位拆分场景动力学。
1. 点云分配到跨帧软槽位。
2. 槽位差分表示潜在动作。
3. 预测未来对象状态并条件化策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32330v1)
- [arXiv](https://arxiv.org/abs/2610.32330v1)

---

<a id='2610.32177v1'></a>
## [$Ψ$-0.6: Adaptive Streams of Thought in a Unified Vision-Language World Model](https://arxiv.org/abs/2610.32177v1)

**Authors:** Khai Loong Aw, Klemen Kotar, Baihan Zhang, Yifan Liu, Lilian Chen, Atlas Kazemian, Rahul Venkatesh, Daniel Yamins

**Published:** 2026-10-23

**Categories:** cs.AI, cs.CV, cs.LG

**Abstract:**

Humans reason about the physical world by flexibly combining different forms of inference as a problem unfolds. Achieving this flexibility is a major open challenge in AI: capabilities often remain separated across specialized models, limiting how they can be composed and jointly adapted to a task. We introduce $Ψ$-0.6, a unified vision-language world model that adaptively composes perception, language, geometry, and simulation within a shared reasoning process we call streams of thought. These streams interleave images, text, and explicit representations of geometry and motion, all of which the model can generate and interpret through a shared token interface. $Ψ$-0.6 surpasses leading open-weight models zero-shot on multi-view spatial reasoning, with further gains from reasoning through geometric intermediates. It also supports counterfactual simulation: generating and interpreting imagined scenes to determine, for example, whether moving an object would obstruct a pathway. Reinforcement learning from task feedback jointly optimizes the selection, generation, and use of intermediate representations, improving visual question-answering accuracy with gains that transfer to unseen benchmarks. $Ψ$-0.6 offers a path toward adaptive physical reasoning -- not only predicting the physical world but also choosing the representations through which to reason.

### 论文解读

#### 摘要翻译

Ψ-0.6是一种统一视觉语言世界模型，试图让同一自回归模型处理图像、相机、深度、光流、点轨迹和文本，并通过提示组合预测、问答、编辑与交互式推理。论文还探索让模型自行决定何时生成中间视觉表征。

#### 方法动机分析

以往视觉模型常按任务拆成深度估计、光流、视频生成和问答系统，结果难共享空间与运动能力。作者的核心假设是把多种视觉表征序列化为同一 token 流，能让模型跨任务复用能力；强化学习则可进一步选择需要的“思维流”。

#### 方法设计详解

统一编码器并非完全共享一个固定输出头：输入和输出模态由提示和专属 token 结构决定，部分任务仍需相应微调数据。RL 学习的自适应思维流只在问答反馈上验证，尚未证明能提升物理规划。

模型使用 8B 自回归 Transformer，32 层、隐藏维度 4096、上下文约 4096 token。RGB、相机参数、深度、光流、点轨迹与文本通过统一多模态 token 接口编码；指针 token 表示模态及空间、时间位置。提示词选择深度估计、视角合成、动作条件生成、问答等输出路径。预训练和微调混合视频、几何和语言序列；后续在 VQA 上用强化学习学习自生成中间表示的触发策略。

#### 方法对比分析

与专用模型相比，Ψ-0.6强调能力组合和统一接口，不是所有单项指标都领先。论文报告针对不同任务可通过提示构造组合推理，但各能力之间的正负迁移尚未完全拆分。适合关注多模态世界模型架构和跨任务表示复用的研究者。

#### 实验分析（精简版）

报告的预训练计算约 1.2×10^23 FLOPs，意味着复现成本显著；更现实的迁移路线是从单项能力和小型提示组合开始。

在 FlowBench-Real 未见测试集上，FlowBench 微调版平均准确率 51.4%，基础模型 37.5%；MindCube 组合提示也有提升。但 RGB+相机条件的几何分数仍低于专用 GeoThinker，部分编辑任务比较使用 oracle mask。论文没有真实机器人控制实验，真实世界泛化结论限于视觉问答和几何任务。

#### 实用指南

重要的工程权衡是统一表示带来的组合能力和序列长度、算力开销之间的取舍；任务微调仍是达到最佳分数的必要环节。

复现要求大规模多模态预训练数据和硬件，报告约 1.2M 监督训练步、2.5T 序列位置、TPU 训练资源。实际迁移时需检查各模态 token 对齐、提示稳定性和任务负迁移；若用于机器人，应额外训练动作闭环、状态估计与安全执行，不能把视觉预测能力直接当控制能力。

#### 总结

核心思想：统一表征，按需组合视觉推理。
1. 多模态内容转为同一序列。
2. 提示指定预测或推理路径。
3. 由反馈学习何时生成中间表示。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32177v1)
- [arXiv](https://arxiv.org/abs/2610.32177v1)

---

<a id='2610.32381v1'></a>
## [UniWorld: A Unified World Model for Real-Time Visual World Simulation](https://arxiv.org/abs/2610.32381v1)

**Authors:** Xin Zhang, Yabo Chen, Shuaiwei Wang, Zixuan Duan, Haibin Huang, Feng Xu, Xuelong Li

**Published:** 2026-10-23

**Categories:** cs.CV

**Abstract:**

Continuous visual world simulation requires coupling evolving scene geometry with temporally persistent appearance. We present UniWorld, a unified framework that integrates depth trajectories and reference appearance within a shared video generation model. The central design maintains visual and geometric histories on a common timeline. Noise-aligned overlapping continuation places generated history at the prescribed sampling noise level and commits only newly generated latents. Rolling depth latent state preserves the corresponding geometric controls, incrementally reads newly available geometry, and keeps the conditioning fixed within each sampling window. Together, these mechanisms ensure exact preservation of latent history while bounding the active depth state by the window size. We evaluate continuous generation, paired history interventions, and long-horizon behavior across diverse scene categories. The same generator supports estimated, native, and on-demand simulator depth, while cached and incremental execution paths produce bitwise-identical outputs. Appearance changes and geometric control branches operate through the same interface without modifying committed history. The measured steady-state DiT execution path achieves real-time generation throughput, demonstrating the efficiency of UniWorld for continuous visual world simulation.

### 论文解读

#### 摘要翻译

UniWorld面向连续视频世界模拟，解决跨窗口生成时历史画面、深度条件和视觉时间相位容易错位的问题。它将上游模拟器提供的深度轨迹与参考图像接入视频扩散模型，逐窗生成连贯外观。

#### 方法动机分析

许多视频生成方法用边界图像或重叠帧连接窗口，但几何条件可能重复读取、相位错位，已生成历史也可能在后续求解中漂移。作者假设只要视频历史、全局噪声和规范化深度共享同一时间索引，并在每一步投影固定历史，就能建立可检验的连续状态契约。

#### 方法设计详解

模块职责分离清楚：参考图负责外观锚点，深度负责每帧几何条件，history projection 固定承诺前缀，rolling cache管理长序列的活跃几何状态。实现以八次模型评估生成每个窗口。

输入是参考图、可选文本及深度序列。模型将每段划为重叠 latent window，固定边界图像和文本条件；对已生成历史在每次 Euler 更新后施加流匹配噪声级的投影，确保网络每步看到一致前缀。深度采用滚动缓存：保留重叠深度，只读取新增长区间，确保跨窗 canonical latent 完全相同。输出 latent 按全局 VAE 相位解码，再裁掉对齐填充。实验用 MiniMax-H3/VDN DMD8，每窗八次速度场评估，8 张 H100。

#### 方法对比分析

与常见边界帧续写相比，本文的主要创新是联合约束几何与历史状态，适合上游模拟器控制的场景。

与只靠边界帧做自回归续写不同，UniWorld把几何和视觉历史一起纳入窗口状态，并明确给出深度缓存和投影规则。它是上游模拟器的可视化生成接口，并不负责动作决策或物理动力学。

#### 实验分析（精简版）

在同一 benchmark 内比较时仍需留意数据构成：十段视频与本地训练清单有交叠，作者另报告排除交叠后的子集结果。

100段30秒视频的整体 PSNR 为 13.3973 dB；模型 960×544 全续接窗达到14.472新帧/秒，满足论文14 FPS目标。24组固定窗消融显示历史投影仅在局部 seam LPIPS 有小幅优势，继续段PSNR与时间误差区间包含零。三种 PyBullet 条件下在线与缓存输出 latent 完全一致，但 fidelity 和场景覆盖仍有限。

#### 实用指南

不同于依赖单张边界帧拼接的方案，本文把历史前缀约束放进每次流场求解；推理速度仅衡量模型核心前向阶段，使用八次 flow evaluation 生成每个窗口。

复现时需对照采样网格、VAE temporal phase、深度归一化方向和首尾裁剪，避免把缓存精确等同生成质量。论文使用8 NFE、video shift 12、音频shift 3；滚动深度活跃状态有界，但全局 latent 仍随时长增长。可应用于模拟器可提供深度的视觉回放或交互预览，不能替代物理仿真或闭环控制。

#### 总结

核心思想：用一致时间状态续生成视频。
1. 模拟器供深度与参考图。
2. 每步投影固定历史并增量读深度。
3. 按全局解码相位输出视频。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32381v1)
- [arXiv](https://arxiv.org/abs/2610.32381v1)

---

<a id='2610.32233v1'></a>
## [TileShare: Training-Free Acceleration of Overlapping Tiled ViT Decoding](https://arxiv.org/abs/2610.32233v1)

**Authors:** Xiangzhong Ye, Zhongshen Li, Shen You, Ziyang Meng, Liang Xu, Junkai Ji, Qiuzhen Lin, Xiangtao Li, Ka-Chun Wong

**Published:** 2026-10-23

**Categories:** cs.CV, cs.LG

**Abstract:**

Fixed-window Vision Transformer (ViT) decoders run at deployment resolution as overlapping tiles: every token in an overlap band is processed by several full decoder instances. In the 2.4B-parameter MiniMax-H3 video decoder, about half of all per-token computation is duplicated. Removing the tiles fails because attention, global tokens and coordinates are bound to them. TileShare keeps each released tile's attention and computes every other operation once per token. The released computation changes at one kind of step only: an overlap token's per-tile results are blended into one shared state with the released weights. Interventions show that this blend costs quality where attention reads the shared states, mainly in the first few layers, which TileShare therefore keeps per tile. Without training, keeping six layers per tile decodes 1.38-1.51x faster than batched and compiled tiling on two GPU types (up to 1.96x with FP8) and is equivalent to the released decoder in PSNR, LPIPS and DISTS on DAVIS videos. On unseen UVG videos with the setting frozen beforehand, it matches PSNR and DISTS and improves LPIPS. TileShare carries over to two further decoders and is again confirmed on unseen data. It is equivalent in PSNR and DISTS on MAGI-1. On the ViTok-v2 image autoencoder at 2K-4K, it matches overlapping tiling in PSNR, improves LPIPS and DISTS and decodes 1.5-1.8x faster. How many layers stay per tile is measured once per model (for decoders, without labels); we report it for 14 decoders and encoders. Optionally, two GPU-hours of self-distillation let MiniMax-H3 share every layer without a loss in PSNR, and a coordinate field trained while the network stays frozen brings fully shared CLIP-based open-vocabulary segmentation to parity with sliding windows on seven benchmarks.

### 论文解读

#### 摘要翻译

TileShare研究部署分辨率下 ViT 解码器的重叠切块冗余：重叠区 token 会被多个 tile 重复计算。论文在保留每个 tile 的注意力窗口、坐标和全局 token 的前提下，让多数逐 token 运算只做一次。

#### 方法动机分析

论文的工程背景是扩散模型逐渐减少去噪步数后，解码器在总时延中的占比变得突出。

直接扩大 tile 会改变训练时的注意力上下文，缩小重叠则让边缘质量较差的预测获得更大权重。作者的核心假设是，重叠冗余中真正需要按 tile 保留的是注意力上下文，而归一化、投影和 MLP 等逐 token 操作可跨 tile 共享；共享引入的误差主要发生在早期层。

#### 方法设计详解

该 pipeline 保持 attention 的查询、键、值集合与位置编码不变，近似只来自 overlap attention 输出融合后状态被下游层共同读取。实际选择 k 时需以目标分辨率、精度和tile布置重新测量。

对每个唯一 token 保存一份状态；每层先统一计算归一化和 QKV，再按原 tile 收集 token，使用各自坐标与 global token 执行注意力。多个 tile 对同一 overlap token 的注意力输出按原 stitching 权重融合，之后的 out-projection 和 MLP 只执行一次。TileShare-k 保留前 k 层完全按 tile 计算，从第 k 层开始共享。实验用模型自身的重建任务选择最小 k，并在冻结设置后做新数据确认。

#### 方法对比分析

与减少重叠或扩大窗口相比，TileShare不改变注意力看到的原始窗口，因此保持了模型训练时的结构。所需 k 随模型而异：MiniMax-H3约4–6层，MAGI-1为3层，ViTok-v2可从0层共享。它适合已经采用滑窗重叠解码、且逐 token MLP 占计算大头的 ViT 解码器。

#### 实验分析（精简版）

它和常见 token pruning 不同：没有删掉内容相似的 token，而是去掉同一个 token在相邻tile里的重复投影与MLP计算。

MiniMax-H3 TileShare-6 比编译批处理切块快1.41–1.51倍，DAVIS上三项指标达到等效标准；冻结后在UVG保持PSNR/DISTS等效并改善LPIPS。快速运动场景的损失更大，四tile角落仍有少量边界差异。方法收益是模型和硬件相关的，并非所有ViT都可直接获益。

#### 实用指南

推理加速来自去除重复矩阵运算，GPU上的内存访存和融合效率会决定实际收益。

部署报告还应同时记录峰值显存和真实端到端请求延迟。

该方法不改变权重、不丢弃 token，也不把一个大窗口送入注意力；其计算图依旧保留原来的 tile 组，只对相同 token 的矩阵运算去重。这样的结构使算法适合已有 tiled decoder 的渐进式优化。

实现依赖预计算 gather 索引、逐层 scatter/blend 和 GPU 编译；论文在RTX PRO 6000及H100测时，也展示FP8 MLP组合可达1.96倍。部署前应分别验证质量和时延，并做目标分辨率 k 扫描；共享整段token会增加内存，不能只看计算量。

#### 总结

核心思想：保留注意力上下文，共享其余计算。
1. 重叠 tile 定义原始注意力窗口。
2. token 运算合并，注意力仍分 tile。
3. 融合重叠结果并按模型选 k。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32233v1)
- [arXiv](https://arxiv.org/abs/2610.32233v1)

---

<a id='2610.32433v1'></a>
## [Tempo-Zero: Recursive Self-Play in Executable Worlds for Video Reasoning](https://arxiv.org/abs/2610.32433v1)

**Authors:** Jingmin Zhu, Yanyan Zou, Xiuyu Li, James Bailey, Dinh Phung, Tao Yu, Yunyun Hou, Jun Liu, Jun Zhou, Sirui Wang, Nan Duan, Jiachen Liu, Qiuhong Ke

**Published:** 2026-10-23

**Categories:** cs.CV

**Abstract:**

Video reasoning requires understanding how events unfold over time, yet videos with accurate temporal annotations are expensive to collect. Self-play offers an alternative by letting a model generate and solve its own tasks without manual annotation. However, existing video self-play methods train on fixed sets of recorded videos, so they cannot alter the underlying event dynamics and rely on noisy pseudo-labels derived from the model's own outputs, such as majority voting, that risk reinforcing shared model errors. To overcome these limitations, we introduce Tempo-Zero, a self-improvement framework that closes a loop of executable environments, execution-verified self-play, and environment evolution, where executing a program renders a video with the ground-truth option and interval of every question. A single policy plays both Questioner and Solver, proposing questions and answering them with timestamped evidence lines, and is trained on execution-grounded rewards instead of model consensus. The Solver's performance steers a curriculum over tasks and difficulty levels, and the policy selects rule changes that enrich the event dynamics of later rounds. Despite being trained solely on simple programs, without supervised fine-tuning or external video data which prior methods typically require, Tempo-Zero outperforms the base models and recent self-play baselines on average across twelve real-world video benchmarks at different model scales, with particularly strong gains on temporal grounding tasks. Project page: https://alchemist-jo.github.io/tempo-zero/

### 论文解读

#### 摘要翻译

Tempo-Zero提出可执行程序驱动的视频自我改进：程序生成视频时同步产生问题答案和事件时间区间，单一视觉语言策略既出题也答题，并根据表现调整训练任务和环境规则。

#### 方法动机分析

现有视频自训练依赖固定真实视频和模型多数投票，伪标签可能把共有错误强化；程序合成虽有精确标签，但任务与事件动力学固定，不能随模型能力变化。作者假设执行轨迹可提供精确答案和时间证据，且改变程序规则即可同步改变事件和标签。

#### 方法设计详解

输入是可执行程序与环境配置，输出是采样视频和可验证问题标签；整个流程由多个互相校验的模块串联。

问题生成和求解使用同一参数快照，Questioner 的 learnability 奖励读取当前 Solver 通过率，但不穿过通过率反传；执行轨迹直接计算证据语句匹配与时间 IoU，避免让语言模型裁判自己的答案。训练用五轮、每轮20步、batch 32和八次rollout。

每个环境程序先执行成视频与事件 trace，所有候选问题需通过两个检查：把真值重新映射到模型实际看到的采样帧，并在事件逆序/置换后检验是否真依赖时间。一个共享策略分 Questioner 和 Solver 两种提示角色：出题需通过格式、答案泄漏和选项合法性程序门控；Solver答题并给事件时间与陈述。奖励由答案正确、证据语句是否对应正确事件、区间重叠构成。题库记录通过率，课程按能力调节任务和难度，策略还选择规则变化并由执行器验证。

#### 方法对比分析

与固定程序数据集不同，Tempo-Zero构成“策略—环境”闭环，任务分配和事件规则都可变；与多数投票式 self-play 不同，标签和证据区间直接来自执行器。方法适用于可程序化生成且事件可验证的时序任务，难以覆盖的自然视觉语义仍是瓶颈。

#### 实验分析（精简版）

基准包含时间定位、长视频理解和视频推理三组；总体分数是跨十二项指标汇总，不能把单项收益解释为所有视频能力都同幅提升。

Qwen3-VL-4B在12个真实视频基准上的总体分数由44.81升到50.81，时间定位平均37.00；4B和8B均优于基础模型，多数主要指标超过视频自训练基线。消融去掉规则选择后总体分数为49.46。训练只使用12类合成环境，实际自然视频迁移效果虽积极，仍不能说明开放场景都适用。

#### 实用指南

完整流程从环境执行开始，经过筛题、双角色采样和奖励计算，再更新共享策略并选择新规则；所有模块围绕同一执行真值闭环。

复现包括12个可执行环境族、每轮课程、Questioner/Solver成组采样与 GRPO 更新；训练配置为5轮、每轮20步、每步32实例、8次Solver rollouts，8 GPU。需特别检查帧采样后的标签重算、静态问题筛除和规则变更预览，才能保证监督仍可见且可复现。迁移自动驾驶可想象为合成交通事件问答/时间定位，但真实车辆交互和安全行为需要另行建模。

#### 总结

核心思想：执行程序生成可验证视频训练。
1. 程序同时产出视频、答案和证据区间。
2. 策略出题答题，执行结果给奖励。
3. 课程和规则随能力演化。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.32433v1)
- [arXiv](https://arxiv.org/abs/2610.32433v1)

---

