time: 20260922

# Arxiv Computer Vision Papers - 2026-09-22

## Table of Contents

1. [BiView-Touch: Learning Bimanual Tactile Representations by Cross-Hand Completion](#2609.23352v1)
2. [Latent Telepathy: Multi-Robot Communication with Self-Supervised Perceptual Latents](#2609.23269v1)
3. [Scenario MPC with STL Specifications and Pareto-Based Feasibility Repair](#2609.23263v1)

---

## Papers

<a id='2609.23352v1'></a>
## [BiView-Touch: Learning Bimanual Tactile Representations by Cross-Hand Completion](https://arxiv.org/abs/2609.23352v1)

**Authors:** Chenxin Liang, Youchen Lai, Chuqiao Lyu, Tianxing Chen, Shoujie Li, Wenbo Ding

**Published:** 2026-09-20

**Categories:** cs.CV, cs.RO

**Abstract:**

Bimanual interaction produces complementary tactile views of the same physical process, yet existing tactile representation learning largely models the two hands independently or combines them only for downstream prediction, leaving their cross-hand relationship unexplored. To exploit this overlooked structure, we introduce BiView-Touch, a tactile-only framework that completes masked target-hand latents from the remaining visible target-hand regions and the synchronized full contralateral hand. A student encoder with a geometry-conditioned directional decoder predicts full-view EMA latent targets, while temporal and layout counterfactuals encourage sensitivity to synchronized and anatomically organized source information. Controlled ablations and source-context interventions show that BiView-Touch learns structured cross-hand dependence on temporally aligned and anatomically organized contralateral tactile context, rather than benefiting from bilateral input alone. On the public HumanTouch dataset, its frozen representations consistently outperform representative self-supervised baselines across low-label settings. With only 5\% downstream labels, BiView-Touch achieves relative balanced-accuracy gains of 7.1\% on bilateral wrist-motion recognition and 14.1\% on force-derived interaction-phase recognition. We further introduce BVT-20, a 20-task bilateral tactile dataset, and demonstrate transfer across recording sessions and pretraining corpora, including transfer to a held-out bimanual task. Our code and dataset details are available on the anonymous project page: https://anonymous.4open.science/w/biview-touch-review-site-050C/.

### 论文解读

#### 摘要翻译
BiView-Touch 是面向双手操作的触觉自监督表征学习框架。它不把左右手独立处理，而是利用同步的对侧手信息，结合目标手可见区域，预测目标手被遮挡区域的潜在表征。方法使用几何条件化方向解码器和 EMA 教师，在 HumanTouch 与新建 BVT-20 上的低标注评估中取得更好的冻结表征，腕部运动识别提升 7.1%，交互阶段识别提升 14.1%。

#### 方法动机分析
双手交互的两路触觉具有互补性：一只手的接触与运动可解释另一只手的局部状态。方法动机是缓解低标注场景下表征缺少跨手时序和空间结构的问题；单手遮挡学习或简单双手拼接没有显式约束同步关系和解剖布局。BiView-Touch 将目标改为跨手条件补全：输入同步的左右手 60 帧窗口，每手 290 个 taxel，11 个解剖区域和 12 个时间 patch；训练时整段遮挡目标手的功能组。

#### 方法设计详解
CrossFormer 维度分段嵌入产生 taxel-time token，再用区域 query 只聚合对应解剖区域。解码器以目标手可见 token 做自注意力，以完整对侧手做 cross-attention，并注入时间、左右手身份及区域质心 Fourier 编码。它输出遮挡区域 latent，而非原始触觉值。EMA 教师对完整输入编码，并减去零输入响应、归一化后作为 stop-gradient 目标。总损失由余弦补全损失、同步排序损失和布局排序损失组成：正确同步上下文须优于时间错位上下文，正确区域布局须优于置换布局。设置为编码维度 192、解码维度 128、4 个 decoder block、学习率 $2\times10^{-4}$、预训练 100 epochs，EMA 动量 0.996–1.0。

#### 方法对比分析
核心差异不是增加一个双手拼接层，而是把跨手关系作为补全任务，并同时建模几何、同步和布局。相对 PatchTST-SSL、Within-hand EMA 及 Raw Dual CNN，该设计更直接检验对侧手是否包含有用条件信息；dense fusion 与 cross-attention 都可作为解码变体。它适合有同步双手触觉、稳定区域映射和低标注下游任务的场景。

#### 实验分析（精简版）
HumanTouch 包含 100 小时、10 个任务；BVT-20 包含 44.2 小时、6,893 sessions 和 22 名参与者。HumanTouch 仅用 5% 标签时，BiView-Touch cross-attention 的 phase/contact/wrist-motion 平衡准确率为 43.88/72.81/65.27%，dense fusion 为 44.27/73.04/65.40%；PatchTST-SSL 为 33.29/66.41/51.67%。力回归 masked region 的 MAE/RMSE 为 0.892/2.388。去掉同步损失使 phase bAcc 从 44.12% 降至 37.13%；错误替换对侧手后腕部运动 bAcc 从 54.93% 降至 19.31%。局限是参与者/任务因素未完全解耦，且尚未验证异构传感器和闭环控制。

#### 实用指南
复现需保留 60 帧同步窗口、每手 290 taxel、11 区域映射、12 时间 patch、功能组遮蔽和时间错位/布局置换负样本。论文报告学习率 $2\times10^{-4}$、100 epochs、EMA 动量 0.996–1.0。代码链接为 anonymous.4open.science 的 biview-touch review site，BVT-20 标为可通过 ModelScope/project page 获取。迁移新传感器时需重建几何区域定义并重训表征或下游头，同时重新检查左右手同步质量。

#### 总结
跨手同步几何补全塑造双手触觉表征。
1. 按区域编码同步双手触觉。
2. 遮蔽目标功能组并读取对侧手上下文。
3. 用几何 decoder 补全 EMA latent。
4. 以补全、同步、布局三项约束训练。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.23352v1)
- [arXiv](https://arxiv.org/abs/2609.23352v1)

---

<a id='2609.23269v1'></a>
## [Latent Telepathy: Multi-Robot Communication with Self-Supervised Perceptual Latents](https://arxiv.org/abs/2609.23269v1)

**Authors:** Howard Wang, Han Zheng, Cathy Wu

**Published:** 2026-09-20

**Categories:** cs.RO, cs.LG, cs.MA

**Abstract:**

In a decentralized multi-robot team under partial observability, the fact that decides a robot's next action is often visible only to a teammate. Existing decentralized methods communicate kinematic information, such as position or planned trajectory, which cannot convey what the teammate perceives. Learned communication in multi-agent reinforcement learning (MARL) can carry perceptual content, but the resulting messages are task-coupled and opaque. We propose Latent Telepathy. Each robot broadcasts the perceptual latent vector it already computes for its own use, the output of an encoder trained with a self-supervised joint-embedding predictive objective, frozen, and shared across the team. A teammate learns to act on it from task reward alone. Because the encoder already runs for perception, the message costs no additional computation and a single compact vector of bandwidth. Because the encoder is frozen before any policy is trained, the message means the same thing to every robot, and the receiving robot is never told what it means. We evaluate Latent Telepathy with a content-controlled protocol in which bandwidth, latency, topology and receiver are held fixed and only the message content varies. Broadcasting the latent lets a navigator avoid an occluded hazard in 99.7% of episodes, matching a noiseless hand-designed message. Position and trajectory messages remain at chance, and the raw camera image, 186 times wider, is less reliable than the compressed latent. The result holds from a discrete gridworld to rendered pixels under continuous velocity control, and the encoder decodes the hazard from a physical robot's camera in 102 of 102 live decisions. We also identify a requirement for porting MARL communication results to continuous control, that the decision a message informs must remain reachable by exploration, and show how to restore it.

### 论文解读
#### 摘要翻译
论文提出 Latent Telepathy，解决部分可观测多机器人中“队友看到的事实无法传给我”的问题。机器人广播为自身感知已计算的自监督潜码，而非位置、轨迹或原始图像；共享编码器预训练后冻结，接收端只凭任务奖励学习读取。这样不增加额外感知计算，并以紧凑向量传递环境内容。

#### 方法动机分析
位置和轨迹只能表达运动学，不能说明某条路是否被障碍堵住；原始图像带宽大且未必易用；端到端 MARL 消息又与任务耦合、语义不透明。论文假设自监督视觉表征仍保留协作所需事实，接收器可以通过奖励学会使用它。当前验证主要针对静止侦察者和二选一堵路事实。

#### 方法设计详解
输入为 64×64×3 RGB 图像。共享编码器由三层卷积（通道 32/64/64，核 8/4/3，步长 4/2/1）和线性层组成，输出 64 维潜码。它先用随机漫游帧进行联合嵌入预测预训练，以方差、协方差约束避免坍缩，随后冻结。机器人广播潜码与相对位置锚点；接收器用单层交叉注意力和掩码池化融合消息，自身策略用 PPO 选择路径，冻结执行器再输出连续速度。训练学习率为 3×10^-3，batch 为 256 个轮次、64 个并行环境；PPO 使用裁剪 0.2、4 个 epoch、γ=0.99、λ=0.95。推理时，消息内容由侦察者当前视角决定，接收者不需要显式标签或人工定义潜码语义。

#### 方法对比分析
在相同带宽、延迟、拓扑和接收器下，潜码区别于 Position/Trajectory 的地方是携带感知内容，区别于端到端通信的是编码器已冻结且不为具体任务专门编码，区别于原图的是压缩和可直接接入策略。它适合已有视觉前端、带宽受限的协作导航，但新传感器域、任务或动作空间通常仍需重训接收器，甚至重训编码器。

#### 实验分析（精简版）
两走廊危险板场景的连续像素控制中，潜码路径优选率为 0.997±0.006、任务成功率为 0.992±0.007，Oracle 为 0.988±0.006；原始图像仅 0.819±0.230，位置/轨迹约为随机。清零、随机噪声或替换消息后降至 0.41–0.56，表明性能来自潜码内容。真实机器人相机测试 102/102 次正确区分危险与安全。限制是场景有限、侦察者静止，且高层决策仍较简化。

#### 实用指南
正文未给出明确代码、模型或数据链接，开源状态未说明。复现需保持感知预训练、冻结编码器、交叉注意力接收器、PPO 路径头和低级执行器，并严格控制通信条件。迁移时应重新检查图像域和潜码分布；新增任务事实或动作空间需重训接收器，并验证连续控制中的可探索性。

#### 总结
核心思想：冻结感知潜码即通信
1. 随机视觉帧预训练共享编码器并冻结。
2. 广播 64 维潜码，接收器用交叉注意力读取内容。
3. PPO 选择路径，冻结执行器输出连续速度。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.23269v1)
- [arXiv](https://arxiv.org/abs/2609.23269v1)

---

<a id='2609.23263v1'></a>
## [Scenario MPC with STL Specifications and Pareto-Based Feasibility Repair](https://arxiv.org/abs/2609.23263v1)

**Authors:** Tianhao Wu, Yiwei Lyu

**Published:** 2026-09-20

**Categories:** cs.RO, eess.SY

**Abstract:**

Temporal logic is a formal language for reasoning about system behaviors over time. Signal temporal logic (STL), in particular, has been used to encode spatio-temporal requirements for control synthesis in multi-agent systems, often under the assumption that agents are cooperative and their dynamics are known. However, real-world multi-agent applications, such as autonomous driving, typically involve stochastic and uncontrollable agents. Recent work explored robust control with worst-case or probabilistic formulations, but remains limited in that it either (1) certifies strict satisfaction of STL constraints without addressing feasibility recovery, or (2) relaxes infeasible constraints with ego-centric objectives. In this paper, we propose a model predictive control (MPC) framework that treats feasibility repair as a Pareto optimization problem to explicitly characterize tradeoffs among agent objectives. We further provide a probabilistic certificate on STL violation rate to formally quantify uncertainty under stochastic and uncontrollable agents. The proposed framework is evaluated on two autonomous driving scenarios. Results show that the framework recovers feasible control with demonstrated safe behaviors.

### 论文解读

#### 摘要翻译
本文针对自动驾驶中存在随机、不可控车辆和行人的多智能体系统，提出结合场景模型预测控制（scenario MPC）与信号时序逻辑（STL）的控制框架。当STL要求因动力学或多方目标冲突而不可行时，方法将修复建模为帕累托优化；同时利用独立验证样本，为未见场景中的STL违反率提供概率保证。作者在两个CARLA自动驾驶场景中验证了可行控制恢复和安全行为。

#### 方法动机分析
普通STL-MPC可能因遮挡、随机行为和动力学限制无解。已有最小空间/时间松弛通常只关注自车或总违规量，忽略救护车、行人等受影响智能体，容易产生集体次优方案。本文的关键假设是可用轨迹样本近似不可控环境，并用独立验证集估计泛化风险；代价是采样和多次MILP求解带来的计算开销。

#### 方法设计详解
输入包括初始状态、预测时域、STL规范、标称控制，以及不可控智能体的联合轨迹样本。系统使用离散动力学和控制仿射近似的运动学自行车模型。第一阶段用N个优化样本求解采样最差情况MPC；若有解，再用独立M个样本验证。验证全部通过时，以至少$1-\beta$置信度保证总体违反率不超过$\epsilon$，样本量满足$M\ge\log(1/\beta)/[-\log(1-\epsilon)]$。

若不可行，则为STL空间谓词加入非负松弛$\xi_k$，使用Big-M把布尔逻辑编码为MILP。方法为每个受影响智能体构造成本$J_i$，用epsilon-约束法把其他目标设为上限，在离散网格上求解多个子问题，得到帕累托候选，最后选择与标称控制偏差最小的方案。推理时，MPC在每个控制周期重新采样/验证并执行选出的控制序列。代价可包含$\|u-u^{nom}\|_1+\|x-x^{nom}\|_1$。核心创新是多智能体目标的帕累托修复，而非单纯增加一个松弛变量。

#### 方法对比分析
Fallback在冲突时默认刹车，可能使救护车追尾；只优化总违规量的最小松弛策略反应慢，可能接近撞击行人。本文通过目标上限网格显式搜索权衡，避免加权和方法依赖手工权重，并通过独立验证补足“训练样本满足”不等于“总体风险可控”的缺口。适用前提是动力学和STL能够被可靠地线性化，且有足够计算资源执行多个MILP子问题。

#### 实验分析（精简版）
场景包括交叉口行人避让和高速公路紧急换道。设置为仿真5 s、规划时域2 s、步长0.1 s、优化样本N=10；$\epsilon=0.05,\beta=0.05$时验证样本M=59，网格密度为5。30次运行的251个通过验证步骤中，250步由10000次蒙特卡洛估计的真实违反率低于0.05，经验一致率99.6%。可行控制平均约需0.1 s和0.15 s；高速公路修复在密度5时耗时44.80 s并求解75个子问题。结果支持安全修复和概率证书，但也显示其修复过程难以满足严格实时要求，且候选存在冗余。

#### 实用指南
论文使用CARLA、CVXPY和Gurobi，未给出可核对的直接代码链接。复现时需按标称控制加入高斯噪声生成环境轨迹，严格分离优化样本与验证样本，实现STL鲁棒度和Big-M逻辑编码，并记录验证参数。迁移到其他机器人或数据集时，要替换动力学、谓词及线性化边界，重新设定N、M、$\epsilon$、$\beta$和网格密度；尤其应先测量多目标MILP在目标智能体数量增加时的耗时。

#### 总结
用帕累托前沿修复不可行STL控制。
1. 场景样本检查STL-MPC可行性。
2. 独立样本给出违反率概率证书。
3. 不可行时松弛空间谓词并构造多智能体目标。
4. epsilon-约束网格搜索帕累托候选。
5. 选最接近标称控制的已验证方案。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.23263v1)
- [arXiv](https://arxiv.org/abs/2609.23263v1)

---

