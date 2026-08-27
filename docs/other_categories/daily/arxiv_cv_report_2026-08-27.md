time: 20260827

# Arxiv Computer Vision Papers - 2026-08-27

## Executive Summary

# ArXiv 计算机视觉日报执行摘要  
**发布日期：2026 年 8 月 26 日**

## 一、总体概览

本期 10 篇论文显示，计算机视觉研究正明显向**具身智能、视觉—语言—动作模型（VLA）、多模态时序建模和可靠性保障**方向集中。主要趋势包括：

1. **从视觉理解走向可执行的世界模型**  
   多篇工作不再局限于图像或视频识别，而是尝试学习“观察—动作—环境变化”之间的关系，包括 Zero-WAM、StreamPI 和 MA-VLA。

2. **机器人操作从单臂、静态任务转向复杂交互**  
   研究重点扩展到接触丰富的操作、多臂协作、开放任务泛化和快速抓取，对真实世界部署更具针对性。相关论文包括 VISTA、MA-VLA 和 Fast Generative Grasping。

3. **多模态模型开始具备更强的时序与代理能力**  
   StreamPI 和 TAU-Agent 分别从流式时序建模、检索增强代理的角度，探索如何让视觉模型处理持续输入、外部知识和复杂决策流程。

4. **生成式视觉模型的缺陷检测与安全问题受到关注**  
   “When Composition Doesn’t Add Up”研究人类如何发现 AI 生成图像中的组合错误；DEFUSE 则关注自监督视觉编码器的后门防御，体现出对生成模型可信性和模型安全的重视。

5. **研究范围从单一算法扩展到交互机制与领域框架**  
   车辆交互中的博弈结构分析，以及 Visual General Intelligence 白皮书，分别代表了对社会化决策和通用视觉智能宏观范式的探索。

---

## 二、值得特别关注的论文

### 1. Zero-WAM：从人类视频学习世界—动作模型

**Zero-WAM**尝试仅利用人类视频进行上下文世界—动作建模，并面向开放式任务泛化。其潜在意义在于：

- 减少对机器人动作标注和真实机器人数据的依赖；
- 探索从被动视频中提取可迁移的环境动力学和任务结构；
- 为“观察人类行为—推断可执行策略”提供统一框架。

如果论文能够有效解决人类动作与机器人动作之间的形态差异和视角差异，它可能对低成本具身学习产生较大影响。

### 2. StreamPI：面向流式输入的视觉—语言—动作建模

StreamPI聚焦于**持续、多模态、时序化输入**，这比传统基于独立观测的 VLA 模型更贴近机器人真实工作环境。其重要性可能体现在：

- 处理连续视频、语言指令和动作反馈；
- 降低长视频或长任务推理中的计算与记忆开销；
- 支持实时决策，而非“看完再行动”的离线范式。

实时性、长期记忆和动作稳定性将是该方向能否落地的关键。

### 3. MA-VLA：多臂协作与组合泛化

MA-VLA将视觉—语言—动作模型扩展到**多机械臂协作**场景，并强调组合泛化。这一方向具有较强的实际价值，因为复杂制造、装配和仓储任务通常需要多个执行体协同完成。

值得重点考察的问题包括：

- 如何分配多臂之间的角色与子任务；
- 如何避免动作冲突和资源竞争；
- 能否泛化到未见过的物体、任务组合和协作关系；
- 模型是否能够处理不同机械臂的动作空间差异。

### 4. VISTA：视觉推断的空间接触注意力

VISTA针对**接触丰富的机器人操作**，从视觉中推断空间接触区域，并将其用于注意力建模。相比仅关注物体级目标的位置，该方向更接近真实操作中的关键难点：

- 接触点、接触面和受力关系；
- 遮挡条件下的局部几何推理；
- 推、拉、插入、旋拧等需要精确接触的动作。

如果其接触表征能够提升跨物体和跨任务泛化，可能成为视觉控制系统的重要中间表示。

### 5. DEFUSE：带生成先验的自监督编码器后门防御

DEFUSE关注自监督视觉编码器的后门攻击问题，并引入生成式先验进行防御。这项工作的重要性在于，自监督编码器通常被广泛复用，一旦预训练模型遭到污染，风险可能传播到多个下游任务。

值得关注的技术点包括：

- 是否能在不依赖大量干净标注数据的情况下发现和消除后门；
- 生成先验是否有助于恢复正常视觉表征；
- 防御是否会显著损害下游性能；
- 对不同架构、数据集和攻击方式的泛化能力。

---

## 三、其他论文的研究价值

- **Fast Generative Grasping via Lie Group-Constrained MeanFlow**  
  将 MeanFlow 类生成方法与李群约束结合，用于快速生成抓取姿态。其亮点在于同时考虑生成效率和机器人位姿的几何结构，适合关注扩散/流模型机器人应用的研究者。

- **TAU-Agent**  
  面向交通异常理解的检索增强代理框架，体现了视觉模型与外部知识、工具调用和多步推理结合的趋势。对智能交通监控和复杂事件解释具有应用价值。

- **When Composition Doesn’t Add Up**  
  研究人类识别 AI 生成图像中组合缺陷的能力。该主题连接了视觉感知、生成模型评估和人类认知，可为图像真实性评测和生成内容检测提供心理学与行为学依据。

- **Visual General Intelligence: A White Paper**  
  作为白皮书，其价值更偏向概念框架、研究议程和领域共识，而非单一算法创新。适合用于了解“视觉通用智能”的定义、能力边界和未来挑战。

- **Choose Your Game Wisely**  
  从博弈论角度分析真实车辆交互，强调驾驶行为不是独立预测问题，而是多主体策略互动问题。对自动驾驶中的意图预测、决策和风险建模具有启发意义。

---

## 四、正在形成的研究方向

### 1. 视频基础模型向世界模型和动作模型演进

未来模型可能不仅回答“视频中发生了什么”，还需要预测：

- 执行动作后环境如何变化；
- 哪些行为是可行的；
- 如何将人类示范迁移到机器人执行；
- 如何在长时间任务中持续规划。

Zero-WAM、StreamPI和MA-VLA共同体现了这一转变。

### 2. 具身模型的实时流式推理

VLA模型正在从离线、单帧、短指令设置，转向：

- 连续视频输入；
- 增量式记忆；
- 低延迟动作输出；
- 在线纠错与闭环控制。

这要求模型结构、缓存机制和训练目标同时面向实时性设计。

### 3. 结构化几何与生成模型结合

Fast Generative Grasping和VISTA表明，机器人视觉研究正在重新强调：

- 李群和位姿几何；
- 接触拓扑；
- 物理可行性；
- 结构化中间表示。

单纯依赖像素相似度或无约束生成，可能难以满足真实操作的精度和稳定性要求。

### 4. 多主体协作与博弈推理

MA-VLA和车辆交互研究共同指向多主体场景。未来视觉系统可能需要同时建模：

- 自身意图；
- 他者意图；
- 角色分工；
- 竞争与合作关系；
- 不确定性下的策略选择。

### 5. 生成模型和基础编码器的可信性

AI 生成内容中的组合错误，以及自监督模型中的后门风险，说明研究重点正在从“模型能否完成任务”扩展到：

- 模型是否可靠；
- 视觉表征是否被污染；
- 生成内容是否具有一致性；
- 人类能否识别模型失败；
- 如何进行可解释和可验证的安全评估。

---

## 五、建议优先阅读全文的论文

### 第一优先级：具身智能与基础模型

1. **Zero-WAM**  
   适合关注世界模型、机器人学习、视频预训练和开放任务泛化的研究者。

2. **StreamPI**  
   适合关注实时 VLA、多模态时序建模和长时程机器人控制的研究者。

3. **MA-VLA**  
   适合研究多机器人协作、组合泛化和语言条件控制的读者。

4. **VISTA**  
   适合关注机器人操作、视觉接触建模和几何感知控制的读者。

### 第二优先级：算法与可靠性

5. **Fast Generative Grasping**  
   如果研究方向涉及生成式策略、抓取规划或 SE(3) 几何建模，值得重点阅读。

6. **DEFUSE**  
   对从事视觉基础模型安全、自监督学习或模型供应链安全的研究者较为重要。

7. **TAU-Agent**  
   适合关注视觉代理、检索增强生成和智能交通应用的读者。

### 第三优先级：趋势与跨领域视角

8. **Choose Your Game Wisely**  
   推荐给自动驾驶、多智能体决策和行为预测方向的研究者。

9. **When Composition Doesn’t Add Up**  
   推荐给研究生成图像评估、人类视觉判断和内容安全的读者。

10. **Visual General Intelligence: A White Paper**  
   适合用于快速建立领域全景和研究议程，但应结合具体技术论文阅读。

---

## 六、总结判断

本期最突出的信号是：**计算机视觉正在从感知模型竞争转向“可行动、可协作、可泛化、可验证”的智能系统竞争。** 其中，Zero-WAM、StreamPI、MA-VLA和VISTA代表了具身视觉从视频理解迈向动作决策和真实交互；Fast Generative Grasping体现了几何约束与生成模型融合；DEFUSE以及 AI 图像缺陷研究则提醒人们，模型安全性和生成内容可信度将成为基础能力的重要组成部分。

若时间有限，建议优先阅读 **Zero-WAM、StreamPI、MA-VLA、VISTA 和 DEFUSE**，它们分别覆盖了开放任务泛化、实时多模态建模、多臂协作、接触操作和模型安全五个最具增长潜力的方向。

---

## Table of Contents

1. [Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization](#2608.26103v1)
2. [Fast Generative Grasping via Lie Group-Constrained MeanFlow](#2608.26076v1)
3. [StreamPI: Streaming Multimodal Temporal Modeling for Vision-Language-Action Models](#2608.26067v1)
4. [TAU-Agent: An Agentic Retrieval-Augmented Framework for Traffic Anomaly Understanding](#2608.25935v1)
5. [When Composition Doesn't Add Up: Humans Identifying Defects in AI-Generated Images](#2608.25933v1)
6. [Visual General Intelligence: A White Paper](#2608.25924v1)
7. [Choose Your Game Wisely: Measuring Game-Theoretic Structures in Real-World Vehicle Interactions](#2608.25917v1)
8. [VISTA: Visually Inferred Spatial ConTact Attention for Contact-Rich Manipulation](#2608.25872v1)
9. [MA-VLA: Multi-Arm Vision-Language-Action Model for Collaboration and Compositional Generalization](#2608.25864v1)
10. [DEFUSE: Generalizable Backdoor Defense for Self-Supervised Encoders with Generative Priors](#2608.25851v1)

---

## Papers

<a id='2608.26103v1'></a>
## [Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization](https://arxiv.org/abs/2608.26103v1)

**Authors:** Jiaming Zhou, Qihang Zhang, Gangwei Xu, Cunxin Fan, Yujie Zhao, Ruilin Wang, Yiming Luo, Shuai Yang, Xing Zhu, Yujun Shen, Junwei Liang, Yinghao Xu

**Published:** 2026-08-26

**Categories:** cs.RO, cs.CV

**Abstract:**

Zero-shot cross-task generalization, where a policy must execute manipulation tasks never seen during training, remains a central challenge in robot learning. In large language models, a novel task can be performed simply by specifying it in the context, without any parameter update. This form of in-context learning (ICL) turns generalization into a problem of task specification. To achieve cross-task generalization, we bring this paradigm to robotic manipulation, and argue that the natural task specification for manipulation is a human video: unlike language, it provides rich visual cues about the intended task evolution. We present Zero-WAM, a causal video-action model that executes unseen tasks by following in-context human video guidance. To address the scarcity of task-rich paired human-robot data, we propose an automatic pipeline that converts task-sampled robot trajectories into semantically matched human videos, yielding HumanGen, a dataset of 74.2K human-robot ICL pairs across 8.6K tasks. For model training, we further introduce an in-context future chunk prediction (IFP) objective that suppresses shortcuts learned from seen tasks and forces the policy to draw task information from the video prompt. On seven unseen tasks in RoboTwin 2.0 simulation, Zero-WAM achieves a 47.0% average success rate, an absolute improvement of 29.5 percentage points over the strongest video-action baseline. In real-world evaluations, it follows human video guidance to generalize to unseen task configurations involving multi-object scenes, long-horizon manipulation, and fine-grained insertion.

**Analysis:**

## 1. 摘要翻译

零样本跨任务泛化要求机器人执行训练阶段从未见过的操作任务，仍是机器人学习的核心难题。大语言模型能够仅通过上下文指定新任务而无需更新参数。本文将这一范式引入机器人操作，并认为人类视频是比语言更自然的任务规范：它提供丰富的视觉线索，展示目标状态及其演化过程。作者提出 **Zero-WAM**，一种因果视频—动作模型，通过上下文人类视频执行未见任务。针对大规模配对人机数据稀缺问题，作者设计自动数据生成流程，将按任务采样的机器人轨迹转换为语义匹配的人类视频，构建包含 **7.42万对样本、8600个任务**的 HumanGen 数据集。同时提出上下文未来分块预测（IFP）目标，抑制模型在已见任务上的捷径学习，迫使策略利用视频提示。在 RoboTwin 2.0 的7个未见任务上，Zero-WAM平均成功率达47.0%，较最强视频—动作基线提升29.5个百分点；真实机器人实验也验证了其在多物体、长时程和精细插入任务上的泛化能力。

## 2. 方法动机分析

**驱动力：**语言难以完整表达空间约束、中间状态和动作顺序，而人类视频直接呈现“物体如何变化、最终变成什么样”。因此，作者将人类视频视为部署时的任务说明，而非传统意义上的动作模仿数据。

**现有痛点：**  
1. 人类视频通常没有可执行机器人动作，难以直接训练控制策略；  
2. 人机成对示范依赖人工采集，任务覆盖有限；  
3. 标准下一时刻预测可仅依赖机器人历史和语言，在已见任务上形成捷径，导致测试时忽略视频提示。

**核心假设：**只要模型学习“人类视频中的任务语义”与“机器人未来视觉状态”之间的对应关系，就能把未见任务的视觉演示迁移为本体相关的机器人动作，而无需复制人的运动轨迹。

## 3. 方法设计详解

### （1）数据构建Pipeline

1. 从 AgiBot、OXE、RoboMIND 等多源机器人数据中按“动作+物体”重新划分任务，并按任务而非原始轨迹频率采样，得到约6000个任务、每轮约40万条轨迹的 **Task-diverse VA**。  
2. 对每条机器人视频，使用VLM解析任务名称、初始状态、状态变化和最终状态。  
3. 根据机器人视频首帧生成图像编辑提示，利用图像编辑模型改变背景、视角、物体实例和摆放位置，得到人类操作场景首帧。  
4. VLM结合首帧和状态变化生成视频提示词，再由视频生成模型合成人类手部操作视频。  
5. VLM评估视频的语义一致性与物理合理性，合格视频与原机器人轨迹配对，形成 HumanGen。

该策略的关键不是动作级对齐，而是**任务语义级对齐**：人类视频提供“要完成什么”，机器人轨迹提供“机器人如何执行”。

### （2）Zero-WAM结构

模型以 Wan-2.2-TI2V-5B 为基础改造成因果视频—动作模型。输入包括机器人历史视频 \(x_{\le i}\)、历史动作 \(a_{\le i}\)、语言指令 \(\ell\)，以及可选的人类视频 \(h\)。模型先预测下一段机器人视频，再根据该预测视频解码动作：

\[
p(x_{i+1},a_{i+1}|history,c)
=p_{vid}(x_{i+1}|history,c)\,
p_{act}(a_{i+1}|history,x_{i+1},c)
\]

视频分支负责预测未来视觉状态，动作分支相当于逆动力学模块，将预测状态转换成可执行动作。采用 Mixture-of-Transformers：视频和动作拥有独立参数，但在共享注意力序列中交互；动作token位于未来视频token之后，因此可以读取预测的视频结果。

在人类视频条件下，人类视频被置于机器人轨迹之前作为前缀记忆。通过高度方向的 RoPE 偏移区分人类视频latent与机器人视频latent，避免两种视觉来源的位置表示混淆。动作分支不直接读取人类视频，而是读取已经吸收任务语义的机器人未来视频。

### （3）IFP目标

普通下一分块预测容易利用局部历史。IFP从当前预测分块的中间表示出发，额外预测多个带时间间隔的未来机器人视频分块。具体使用主视频Transformer多个中间层表示，经MLP融合为 \(\phi\)，再由多个IFP模块预测未来目标：

\[
L_{IFP}=\sum_k w_k L_{fm}(x_{j_k};\phi,history,\ell)
\]

IFP模块**不能直接访问人类视频**，只能使用主分支与人类视频交互后形成的 \(\phi\)。这保证辅助损失真正监督主分支编码视频提示中的长期任务信息，而不是训练一个独立的视频条件预测器。推理时移除IFP模块。

## 4. 方法对比与创新

Zero-WAM区别于语言条件VLA：任务接口从文字扩展为视觉演示；区别于传统视频模仿：它不要求人机动作或视角严格对应；区别于测试时训练方法：部署时不更新参数、不构造记忆适配器。

主要创新包括：  
1. 自动生成大规模语义匹配人机ICL数据；  
2. 通过任务级重采样提升机器人预训练的任务多样性；  
3. 用IFP解决“忽略上下文视频”的训练捷径问题。  

适合多物体、长时程、空间关系复杂、语言难以精确描述的桌面操作任务。

## 5. 实验分析

作者在RoboTwin中按任务划分43个已见任务和7个未见任务，并进行真实双臂Franka实验。代表性结果是：仿真平均成功率47.0%，显著高于LingBot-VA的17.45%；去掉IFP后平均性能降至28.55%。真实实验中，Zero-WAM在多物体顺序操作和双桌腿插入上也优于语言基线。

**优势：**任务接口直观、无需未见任务机器人示范、适合细粒度和长时程指令。  
**局限：**人类视频主要由生成模型合成，存在语义或物理伪影；依赖高质量VLM、图像和视频生成器；实验仍以桌面静态操作为主，长时程能力有限。

## 6. 实用指南

论文提供项目主页，但文中未明确说明代码和完整数据是否公开。复现需准备：多源机器人视频—动作数据、任务级重采样、VLM状态解析、图像编辑和视频生成模型。关键配置包括：视频/动作隐藏维度3072、30层Transformer、IFP未来分块数 \(K=4\)、时间步长 \(s=2\)、权重为(0.5,0.25,0.15,0.15)，预训练数据采样比例VA:HumanGen=1:5，学习率 \(10^{-4}\)。迁移到其他任务时，应保持“人类视频表达任务语义、机器人数据提供动作监督”的配对机制，并针对新机器人补充少量本体适配数据。

## 7. 总结

**核心思想：**用人类视频指定机器人未见任务。

**速记版Pipeline：**  
1. 按任务均衡采样机器人轨迹；  
2. 自动生成语义相同但外观不同的人类操作视频；  
3. 训练模型先预测机器人未来画面，再生成动作；  
4. 用多步未来预测迫使模型理解视频提示；  
5. 测试时输入一段人类视频即可执行新任务。

**Key Findings:**

- In large language models, a novel task can be performed simply by specifying it in the context, without any parameter update.
- We present Zero-WAM, a causal video-action model that executes unseen tasks by following in-context human video guidance.
- To address the scarcity of task-rich paired human-robot data, we propose an automatic pipeline that converts task-sampled robot trajectories into semantically matched human videos, yielding HumanGen, a dataset of 74.2K human-robot ICL pairs across 8.6K tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.26103v1)
- [arXiv](https://arxiv.org/abs/2608.26103v1)

---

<a id='2608.26076v1'></a>
## [Fast Generative Grasping via Lie Group-Constrained MeanFlow](https://arxiv.org/abs/2608.26076v1)

**Authors:** S. Talha Bukhari, Yi Wei, Ruiqi Ni, Zachary Kingston, Aniket Bera

**Published:** 2026-08-26

**Categories:** cs.RO

**Abstract:**

Grasp synthesis is a core task in robotic manipulation, for which the solution typically forms a multimodal distribution rather than a point estimate. Generative robotic grasping aims to learn this distribution with deep generative models such as diffusion and flow-based approaches. The iterative nature of such generative models makes them flexible and generalizable; however, multi-step sampling impedes the time-critical operation required in robotics. We devise an approach to fast generative grasping based on MeanFlow on the product Lie group $\mathcal{G} = \mathrm{SO}(3) \times \mathbb{R}^3$. The training objective couples a purely algebraic semigroup consistency condition with Riemannian Conditional Flow Matching on $\mathcal{G}$ that anchors the average velocity to the data distribution. The resulting Lie Group-constrained MeanFlow formulation samples reliable grasps in $\leq 5$ network evaluations, matching the grasp generation performance of state-of-the-art diffusion and flow-based models on the ACRONYM dataset at millisecond-scale inference latency (up to $39\times$ speed-up). We further demonstrate that the approach directly translates to real-world robotic grasping without additional training or domain adaptation, exhibiting robust grasp synthesis under observation noise.

**Analysis:**

## 1. 摘要翻译

抓取合成是机器人操作的核心任务，其解通常表现为多峰分布而非单一姿态。生成式机器人抓取利用扩散模型或流模型学习该分布，但迭代采样过程难以满足机器人实时操作需求。本文基于乘积李群 \(\mathcal G=SO(3)\times\mathbb R^3\) 提出快速抓取生成方法 GraspMF。训练目标将纯代数的半群一致性条件与黎曼条件流匹配结合，使平均速度由数据分布进行锚定。所得模型仅需不超过5次网络评估，即可生成可靠抓取；在ACRONYM数据集上，其性能匹配先进扩散和流模型，同时达到毫秒级推理速度，最高加速39倍。实验还表明，该方法无需额外训练或域适配即可迁移到真实机器人，并能在观测噪声下保持稳定抓取生成。

## 2. 方法动机

**驱动力与痛点：**扩散模型通常需要几十至上百步，Flow Matching虽更快，但低步数时对瞬时速度场进行欧拉积分会产生较大截断误差，导致姿态偏离有效接触流形。Consistency/Shortcut方法虽能加速，却主要学习自洽目标，未必与真实输运场一致。

**核心假设：**若直接学习一个时间区间上的“平均速度”，并要求不同区间的跳跃满足流映射半群性质，那么模型可以用大步弦式更新近似完整输运过程，而不是逐步积分瞬时速度。

## 3. 方法设计详解

### Pipeline

1. **李群表示。**将抓取姿态表示为 \(H=(R,p)\)，其中 \(R\in SO(3)\)、\(p\in\mathbb R^3\)。采用直积群而非标准半直积 \(SE(3)\)，从而获得解耦的旋转/平移度量与双不变结构。  
2. **构造训练路径。**从先验 \(H_0\) 和真实抓取 \(H_1\) 采样，通过李群测地线插值得到  
\[
H_t=\mathrm{Exp}_{H_0}(t\mathrm{Log}_{H_0}(H_1)).
\]
先验为SO(3) Haar均匀分布与平移高斯分布的乘积。  
3. **端点预测。**网络输入当前姿态 \(H\)、起止时间 \((s,t)\) 及点云条件，直接预测干净抓取 \(\hat H_1=X_\theta(H,s,t)\)，而不是直接预测速度。  
4. **闭式转换为平均速度。**利用
\[
\bar u_\theta(H,s,t)=\frac{1}{1-s}\log(H^{-1}\hat H_1),
\]
得到区间平均速度，并构造更新
\[
\Phi_\theta(H,s,t)=H\exp((t-s)\bar u_\theta).
\]
直观上，这是沿当前姿态到预测终点的旋转测地线和平移线段前进相应比例。  
5. **数据锚定。**在时间对角线 \(s=t\) 上，将模型预测端点与真实 \(H_1\) 的条件流速度对齐，防止仅靠一致性损失收敛到“恒等映射”这一无意义解。  
6. **半群一致性。**要求一次从 \(s\) 跳到 \(t\)，等价于先跳到中间时刻 \(r\)，再跳到 \(t\)：  
\[
\Phi_\theta(H,s,t)\approx
\Phi_\theta(\Phi_\theta(H,s,r),r,t).
\]
训练时冻结两步组合结果作为目标，仅使用网络前向、群指数和对数，无需协变导数、\(\mathrm{dexp}^{-1}\)或高方差微分项。总损失为流匹配锚定损失与半群损失的加权和。  
7. **推理。**从先验出发，在时间区间上均匀划分为 \(T\) 段，每段调用一次网络并执行闭式李群更新。\(T=1\)时直接输出单步抓取，\(T=5\)时获得更稳定结果。

### 模型结构

采用轻量化点云网络：VNN编码器提取具有旋转等变性的物体描述；姿态通过随刚体变换的关键点映射为特征；两个时间变量使用随机傅里叶特征编码。输出为9维旋转矩阵参数和3维平移，旋转矩阵通过SVD投影到 \(SO(3)\)。另加入SDF辅助回归，使共享特征显式感知物体几何。

## 4. 对比与创新

与扩散模型的根本区别是：扩散依赖随机反向去噪，Flow Matching依赖瞬时场积分；GraspMF学习的是区间平均输运并直接预测终点。其主要创新是将**李群上的代数半群一致性**与**条件流匹配数据锚定**结合，既保证几何约束，又避免微分型MeanFlow目标的不稳定。适合姿态位于流形、需要多峰采样且强调低延迟的抓取、操作策略和机器人规划任务。

## 5. 实验分析

作者在ACRONYM、Isaac Gym、部分点云和Franka真实机器人上验证。核心结论是：GraspMF在5步时ID/OOD成功率达87.40%/71.73%，延迟15.5 ms；单步仍保持81.11%/66.34%，延迟仅6.3 ms。去除半群损失后性能显著下降，说明一致性约束是低步数保持接触可行性的关键。局限包括：旋转对数在反平行附近存在分支问题；方法依赖测地线局部可逆性，且部分观测下分布覆盖指标下降；实验也未充分讨论更复杂接触或双臂任务。

## 6. 实用指南

文中未明确提供代码仓库，开源情况需另查作者主页或论文页面。复现关键点包括：使用 \(SO(3)\times\mathbb R^3\) 运算及稳定对数分支；SVD旋转投影；\(\lambda_{\rm CFM}=1\)，半群权重从0线性升至1；锚定时间偏向接近1，并将 \(1/(1-t)\) 截断；半群尾部使用 \(w(r)\) 抑制噪声；训练中加入SDF监督。迁移到其他李群时，只需替换群的指数、对数、测地线和左平凡化操作，并重新设计姿态输出与几何辅助目标。

## 7. 总结

**核心思想：**在李群上学习可自洽的平均输运。

**速记版Pipeline：**

1. 用旋转加平移表示抓取姿态；  
2. 在先验姿态与真实抓取之间建立测地线路径；  
3. 网络直接预测最终干净抓取；  
4. 用端点差得到平均移动，并约束不同时间区间前后一致；  
5. 通过1—5次弦式跳跃生成最终抓取。

**Key Findings:**

- The resulting Lie Group-constrained MeanFlow formulation samples reliable grasps in $\leq 5$ network evaluations, matching the grasp generation performance of state-of-the-art diffusion and flow-based models on the ACRONYM dataset at millisecond-scale inference latency (up to $39\times$ speed-up).

**Links:**

- [PDF](https://arxiv.org/pdf/2608.26076v1)
- [arXiv](https://arxiv.org/abs/2608.26076v1)

---

<a id='2608.26067v1'></a>
## [StreamPI: Streaming Multimodal Temporal Modeling for Vision-Language-Action Models](https://arxiv.org/abs/2608.26067v1)

**Authors:** Zhe Liu, Jinghua Hou, Yuxiang Lu, Zhenya Yang, Xianzhe Fan, Junwei Luo, Junyi Li, Ruihua Han, Zhi Hou, Hengshuang Zhao

**Published:** 2026-08-26

**Categories:** cs.CV

**Abstract:**

Vision-Language-Action (VLA) models have demonstrated effectiveness in robot manipulation, yet state-of-the-art models such as pi0.5 operate under a single-frame paradigm, limiting their ability to retain past observations and develop precise spatial perception. In this paper, we propose StreamPI, a streaming multimodal temporal modeling framework that equips single-frame VLA with temporal reasoning capability without introducing any additional parameters. One core design is instruction-anchored temporal modeling. It treats each (visual observation, language instruction) pair as an atomic temporal unit: bidirectional attention within each pair enables cross-modal fusion, while causal attention across pairs preserves autoregressive streaming inference. This ensures the language instruction serves as a persistent semantic anchor throughout task execution. To bridge the gap between synchronous training and asynchronous real-robot deployment, we introduce a andom-interval streaming training strategy: a proper inter-frame interval (e.g., every 3 frames) enables faster and smoother action execution. Beyond this, randomizing the interval further improves robustness to frame-timing perturbations, supporting asynchronous deployment in practice. Furthermore, by leveraging the length extrapolation capability of the LLM backbone, StreamPI seamlessly inherits pretrained single-frame weights and supports flexible single-frame and multi-frame inference. Experiments on real-robot tasks spanning memory-dependent and precise perception scenarios, as well as the simulation benchmark LIBERO, demonstrate that StreamPI outperforms pi0.5 across diverse tasks.

**Analysis:**

# 1. 摘要翻译

视觉-语言-动作（VLA）模型已在机器人操作中展现出良好效果，但 π0.5 等先进模型采用单帧输入范式，难以保留历史观测，也限制了精确空间感知能力。本文提出 StreamPI，一种无需增加参数的流式多模态时序建模框架，使单帧 VLA 具备时序推理能力。其核心是**指令锚定的时序建模**：将每个“视觉观测—语言指令”对视为原子时序单元，在单元内部采用双向注意力进行跨模态融合，在单元之间采用因果注意力保持自回归流式推理，使语言指令在整个任务执行过程中持续发挥语义锚点作用。为弥合规则采样训练与真实机器人异步部署之间的差异，作者提出随机间隔流式训练，通过随机化帧间隔提升模型对时间扰动的鲁棒性。此外，借助 LLM 主干的长度外推能力，StreamPI 可直接继承单帧模型权重，并支持单帧与多帧推理。真实机器人和 LIBERO 实验表明，StreamPI 在记忆依赖和精确感知任务上均优于 π0.5。

# 2. 方法动机分析

**驱动力：**机器人当前动作往往不仅取决于当前画面，还取决于物体运动、遮挡前状态和任务执行阶段。单帧 VLA 无法记忆历史，也难以从多帧变化中获得深度、运动和几何信息。

**现有痛点：**窗口式多帧方法需重复编码全部历史帧，计算和延迟随帧数增长；简单拼接视觉帧会稀释语言指令，产生“指令遗忘”；训练中的固定采样间隔与真实机器人的异步观测不匹配；额外视频编码器还可能破坏 π0.5 已有的视觉语言表征。

**核心假设：**只要让每一帧都与指令绑定，并让历史信息以缓存形式逐步累积，就能在不改动模型参数的前提下获得稳定的时序记忆与更强空间感知。

# 3. 方法设计详解

## 3.1 整体流程

1. **输入构造：**时刻 \(t\) 输入三路图像（前视、左右腕部相机）和语言指令 \(l_t\)，形成原子单元  
   \[
   u_t=(V_t,l_t).
   \]

2. **单元内融合：**图像 token 与文本 token 在同一单元内使用双向注意力，彼此可以充分交互，得到融合表示 \(h_t\)。这一步保留了 π0.5 原有的跨模态理解能力，尤其避免视觉 token 无法读取指令的问题。

3. **跨时序聚合：**多个单元按时间顺序排列。当前单元可以访问历史单元，但历史单元不能访问未来单元，即采用块状因果注意力。于是模型既能利用过去观测，又不会发生训练时未来信息泄漏。

4. **动作生成：**当前时刻输出表示 \(o_t\) 条件化动作专家/flow-matching 模块，生成动作块，而非仅生成单个动作。

5. **流式缓存：**首次推理编码当前单元并写入 KV-cache；之后只编码新到达的图像—指令单元，让其查询历史缓存。缓存达到最大长度 \(T\) 后，通过滚动或清空机制维持有限上下文。

## 3.2 关键设计

- **指令锚定：**不是只在序列开头放一次指令，而是将同一任务指令复制到每个时间单元中。这样每帧视觉信息都具有明确任务语义。
- **双向—因果混合掩码：**单元内双向、单元间因果；这比全因果注意力更适合图文融合，比全双向注意力更符合在线部署。
- **无参数扩展：**仅改变输入组织方式、位置编码长度和 attention mask，不增加视频编码器或记忆模块参数，因此可继承 π0.5 权重。
- **随机间隔训练：**训练时从区间 \([3,7]\) 随机采样帧间隔，并随机屏蔽最早若干帧，模拟真实部署中的不同帧率、延迟和不完整历史。

需要注意，“计算量恒定”成立的前提是缓存大小受 \(T\) 限制；若缓存无限增长，注意力开销仍会增加。论文附录中的延迟也显示，\(T\) 越大推理时间略有上升。

# 4. 方法对比分析

**本质区别：**窗口式方法每次重新处理整段视频；StreamPI 只处理新帧并复用历史 KV。普通多帧拼接以“视觉序列”为中心，StreamPI 以“图像—指令对”为时序基本单位。

**创新点：**①指令锚定的原子时序单元；②单元内双向、单元间因果的掩码设计；③面向异步机器人的随机间隔训练；④无需新增参数即可把单帧 VLA 改造成流式模型。

**适用场景：**遮挡记忆、动态目标拦截、长序列操作、精细插入和多视角几何感知。对于静态、单帧已足够的任务，收益可能有限。

# 5. 实验分析

作者在真实 PiperX 机器人、LIBERO 和 CALVIN 上验证。代表性结果是：真实任务中，滚动瓶抓取成功率由 26.7% 提升至 63.3%，贝壳游戏由 46.7% 提升至 80.0%；LIBERO 中 \(T=5\) 的平均成功率为 98.3%，高于 π0.5 的 96.9%。

主要优势是改动小、无需新参数、支持流式缓存，并显著增强记忆和精确感知。主要局限是训练仍需同时加载多帧，超长时序代价高；随机间隔也不能完全解决极端异步问题。

# 6. 实用指南

论文提供项目主页和视频，但给定文本未明确声明代码已开源，复现需自行实现。关键设置：\(T=3/5\)，训练间隔随机采样 \([3,7]\)，仿真推理间隔为 5，真实机器人间隔随机采样；保持 π0.5 优化器和学习率，对原模型进行全量微调。实现重点是正确构造块状 attention mask、扩展位置编码、缓存每层 Key/Value，并保证缓存顺序与时间戳一致。该框架可迁移到视频语言模型、导航、驾驶和长时序预测任务，需将动作头替换为对应任务输出，并重新设计时序采样策略。

# 7. 总结

**核心思想：**让每帧视觉都带着任务指令进入流式记忆。

**速记版 pipeline：**

1. 每次接收新图像，并与任务指令组成一对。  
2. 让图像和指令充分互相理解。  
3. 新信息读取历史缓存，但不能偷看未来。  
4. 根据当前与历史信息生成动作块。  
5. 保存新信息，按随机时间间隔持续执行。

**Key Findings:**

- Vision-Language-Action (VLA) models have demonstrated effectiveness in robot manipulation, yet state-of-the-art models such as pi0.5 operate under a single-frame paradigm, limiting their ability to retain past observations and develop precise spatial perception.
- In this paper, we propose StreamPI, a streaming multimodal temporal modeling framework that equips single-frame VLA with temporal reasoning capability without introducing any additional parameters.
- To bridge the gap between synchronous training and asynchronous real-robot deployment, we introduce a andom-interval streaming training strategy: a proper inter-frame interval (e.g., every 3 frames) enables faster and smoother action execution.
- Experiments on real-robot tasks spanning memory-dependent and precise perception scenarios, as well as the simulation benchmark LIBERO, demonstrate that StreamPI outperforms pi0.5 across diverse tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.26067v1)
- [arXiv](https://arxiv.org/abs/2608.26067v1)

---

<a id='2608.25935v1'></a>
## [TAU-Agent: An Agentic Retrieval-Augmented Framework for Traffic Anomaly Understanding](https://arxiv.org/abs/2608.25935v1)

**Authors:** Yuqiang Lin, Yan Shi, Sam Lockyer, Harish Tayyar Madabushi, Adrian Evans, Wenbin Li, Yinhai Wang, Nic Zhang

**Published:** 2026-08-26

**Categories:** cs.CV, cs.AI

**Abstract:**

Traffic Anomaly Understanding (TAU) requires models and systems to detect, reason about, and explain anomalous events in transportation videos. To address this challenge, we propose TAU-Agent, an agentic retrieval-augmented framework for traffic anomaly understanding. Given a task query, a central retrieval agent orchestrates two visual perception tools, namely a Video Captioning Tool and an Open-Vocabulary Tracking Tool, to retrieve and select query-relevant evidence, including captions, temporal intervals, and object trajectories. The selected evidence, together with sampled video frames and the input query, is provided to a supervised fine-tuned vision-language model for final reasoning and answer generation. We evaluate TAU-Agent on both the in-domain and the out-of-domain benchmarks from the AI City Challenge 2026. TAU-Agent achieves scores of 0.6779 on Track 3, 0.3998 on Track 7, and 67.9275 on Track 8, ranking second, twelfth, and fifth, respectively. Code is available at: https://github.com/siri-rouser/TAU-Agent.

**Analysis:**

## 1. 摘要翻译
交通异常理解（TAU）要求系统从交通视频中检测、推理并解释异常事件。作者提出 TAU-Agent，一种智能体式检索增强框架：中央检索智能体调用视频描述工具和开放词汇跟踪工具，选择与问题相关的描述、时间区间和目标轨迹；随后将这些证据、采样帧及问题输入经过监督微调的视觉语言模型，生成最终答案。在 AI City Challenge 的 Track 3、7、8 上分别取得 0.6779、0.3998 和 67.9275 的成绩。

## 2. 方法动机
**驱动力**：同一视频包含多个异常、正常事件和大量无关内容，而问题只关注特定目标、交互或时间段；有效证据在空间和时间上都很稀疏，均匀采样容易漏掉关键瞬间，也会引入冗余信息。  
**现有痛点**：端到端视频模型通常直接处理固定采样帧，难以同时完成事件定位、目标跟踪、因果解释和多种答案格式预测；传统异常检测方法又往往只关注异常分数或时间定位。  
**核心假设**：先依据问题检索和压缩相关证据，再让 VLM 推理，比让模型直接浏览整段视频更可靠。

## 3. 方法设计详解
### 整体流程
1. **问题解析**：主 RAG Agent识别问题涉及的事件、目标、交互、异常类型和时间线索。  
2. **视频描述**：视频切成不重叠的 2 秒片段，每片段以 2 FPS 采帧，由 Gemini 生成局部描述；再按时间顺序生成全视频摘要，并用全视频均匀采样的 4 帧生成全局场景描述。  
3. **时间证据筛选**：智能体结合问题与局部描述，确定候选相关区间，并选择相关描述。  
4. **目标证据检索**：若问题需要对象级信息，调用开放词汇跟踪工具。COCO类别及车辆细粒度属性走 YOLO 检测，再用车型、颜色分类器过滤；非 COCO 类别由 GroundingDINO 根据文本检测。所有检测结果用 ByteTrack 跨帧关联，轨迹按 1 FPS采样，最多保留20个观测，每项包含帧号、框坐标、类别和置信度。  
5. **证据精炼与迭代**：Agent联合分析问题、描述和轨迹，重新确定帧范围、保留证据并赋予相关性分数；若证据不足，可再次调用工具。最终保留得分最高的5段描述和5条轨迹。  
6. **慢快采样与回答**：全视频以2 FPS保留上下文，相关区间以4 FPS密集采样；帧、文本证据和问题共同输入 LoRA 微调的 Qwen3-VL-8B。  
7. **可选跨问题上下文**：Track 3中，从同视频相关问题提取事实信息、候选信息和时间范围，辅助当前问题；不同任务则选择性启用，避免错误偏置。

### 关键设计逻辑
该方法不是简单增加视觉工具，而是把“检索什么、何时检索、是否需要跟踪、哪些证据最终送入模型”交给任务自适应的 Agent，从而实现查询条件下的空间—时间证据压缩。

## 4. 方法对比与创新
本质区别在于：主流方法多采用固定帧采样和单模型统一推理，TAU-Agent则将感知、证据选择和回答解耦，并允许动态工具调用。创新主要包括：  
1. 描述检索与开放词汇轨迹检索的协同；  
2. 面向问题的时间范围和证据排序；  
3. 训练阶段用答案及 CoT 验证检索证据，降低噪声；  
4. 针对相关问题设计可开关的上下文代理。  
适合长交通视频、异常事件稀疏、问题类型多样且需要解释或定位的场景。

## 5. 实验分析
作者在 Track 3（域内）、FETV（鱼眼视频）和 PSI-VQA（车载视角、行人意图）验证泛化能力。代表性结论是：Track 3排名第二，说明查询相关证据检索有效；PSI-VQA排名第五，且开放问答线索 F1 最优，表明该框架能迁移到不同视角和任务。  
**优势**：证据稀疏场景更高效，支持目标级、时间级和文本级联合推理。  
**局限**：依赖外部大模型和多次工具调用；检索错误会传播；对鱼眼图像和结构化 JSON 预测适配不足；跨问题上下文可能造成答案偏置。

## 6. 实用指南
代码已开源：`github.com/siri-rouser/TAU-Agent`。复现重点是：构建2秒片段描述和全局描述；部署 YOLO/GroundingDINO+ByteTrack；实现 Agent 的工具调用、证据排序及慢快采样；使用 LoRA 微调 Qwen3-VL。关键设置包括最大100帧、2/4 FPS采样、LoRA \(r=128,\alpha=256\)、dropout 0.03、训练2轮、学习率 \(5×10^{-5}\)。迁移到其他任务时，只需替换问题模板、检测类别、输出格式和任务专用提示；若无相关问题集合，应关闭跨问题代理。

## 7. 总结
**核心思想**：先检索相关证据，再进行视频推理。

**速记版 pipeline**：  
问题拆解 → 描述视频并找时间段 → 按需跟踪目标 → 筛选并验证证据 → 联合帧与证据生成答案。

**Key Findings:**

- To address this challenge, we propose TAU-Agent, an agentic retrieval-augmented framework for traffic anomaly understanding.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.25935v1)
- [arXiv](https://arxiv.org/abs/2608.25935v1)

---

<a id='2608.25933v1'></a>
## [When Composition Doesn't Add Up: Humans Identifying Defects in AI-Generated Images](https://arxiv.org/abs/2608.25933v1)

**Authors:** Ruoqi Hu, Chulin Zhao, Jiashuo Chang, Ramon Ruiz-Dolz, Hanhe Lin

**Published:** 2026-08-26

**Categories:** cs.CV, cs.AI

**Abstract:**

*Chulin Zhao and Ruoqi Hu contributed equally to this work.   State-of-the-art text-to-image (T2I) models exhibit pronounced and systematic defects when prompts involve intricate compositional factors such as multiple entities and multiple attributes. In this paper, we investigate how humans identify such defects. Specifically, we manually select 651 reference images from the four categories of people, hand, object, and scene that exhibit complex compositional characteristics, from which prompts emphasizing compositional factors are derived by manually editing ChatGPT-generated prompts. We then feed the prompts into three selected T2I models to generate AI images and conduct a comprehensive subjective study to identify their defects. For each image, 29 participants provide multi-label assessments specifying defect types and locations. The study yields the compositional AI-generated image defect (CO-AID) dataset, including reference images, prompts, AI-generated images, and information on defect locations and types. Experimental results show that training a deep model on CO-AID can both predict defects in AI-generated images and optimize AI image generation, demonstrating its usability and effectiveness. The database and supplementary materials are available at: https://github.com/Future-IQA/CO-AID .

**Analysis:**

## 1. 摘要翻译

当前先进的文生图（T2I）模型在涉及多实体、多属性等复杂构图因素的提示词下，会产生明显且具有系统性的缺陷。本文研究人类如何识别这些缺陷。作者从人物、手部、物体和场景四类中人工选取651张具有复杂构图特征的真实参考图，并基于ChatGPT生成、人工修改构图提示词，再输入Midjourney、Imagen和FLUX生成AI图像。29名参与者对每张图进行多标签评价，标注缺陷类型及位置，由此构建CO-AID数据集。实验表明，利用CO-AID训练深度模型，既能预测AI图像缺陷，也能辅助图像修复和生成优化。

## 2. 方法动机分析

**驱动力：**现有T2I模型在单主体、单属性生成上表现较好，但面对多人、多物体、属性绑定、遮挡、空间关系和交互时容易失效。作者希望研究的不只是“图像整体好不好”，而是人类如何定位并解释局部构图错误。

**现有痛点：**已有构图评测主要关注全局忠实度或偏好分数，缺少细粒度缺陷位置与类别；异常人体研究又多针对简单场景，无法覆盖复杂实体关系。

**核心假设：**复杂构图缺陷具有可被人类稳定感知的全局和局部模式；将这些模式系统标注后，可以训练缺陷预测模型，并反过来指导图像修复。

## 3. 方法设计详解

### Pipeline

1. **参考图构建：**从Pexels和Unsplash选取651张真实图，分为people、hand、object、scene四类，数量分别为166、161、161、163；object进一步含动物、机器、乐高、食物和杂货，scene含现代、自然、抽象场景。手部被单独划分，因为其结构复杂且常涉及手—物交互。  
2. **提示词生成：**先让ChatGPT描述参考图，再人工修改，确保提示词明确包含：多个实体及属性、二维/三维空间关系（相对位置、遮挡）、实体与环境或物体之间的交互。  
3. **AI图像生成：**预实验比较多个模型后选用Midjourney、Imagen和FLUX。651个提示词随机分配给三个模型，各生成一张图；样本划分为11张训练示例、40张pilot图和600张正式实验图。  
4. **人类标注：**参与者经过介绍、11张训练图和正式实验三个阶段。若无明显缺陷，选择“No noticeable defect”；否则可进行全局和/或局部标注。全局标注包括不自然风格、模糊/细节缺失、异常文字、常识违背，以及大量异常实体、实体交互异常和空间关系异常。局部标注按face、hair/fur、hand、body、object分类，手部还细分结构、手指数目、姿态和指甲异常等。局部缺陷通过点击位置并选择原因记录，也允许自由文本补充。  
5. **数据清洗：**以标注耗时、重复图一致性和与多数意见的一致性衡量参与者可靠性，剔除6名异常参与者，最终保留23人，共获得18,906条标注。  
6. **缺陷预测与修复验证：**将局部点击转换为缺陷热图，微调TranSalNet进行缺陷显著性预测；同时用GPT-Image-1进行对比检测，再把预测缺陷图作为空间引导输入GPT-Image-1，执行缺陷修复。

### 模型协同

T2I模型负责制造复杂构图样本，人类提供缺陷位置和语义标签，TranSalNet学习“哪些区域容易被人认为有问题”，GPT-Image-1则根据缺陷区域执行修复。论文没有提出新的生成网络或核心数学公式，主要创新在于数据采集、标注体系和“人类感知—预测—修复”闭环。

## 4. 方法对比与创新

其本质区别是：从传统的全局质量/文本忠实度评估，转向**面向复杂构图的局部缺陷定位与原因解释**。创新包括：首次系统定义构图缺陷层级；建立包含参考图、提示词、生成图、位置和类别的CO-AID；用人类标注监督缺陷预测，并验证缺陷图可指导修复。适合T2I质量评估、生成模型诊断、局部修复和人类反馈训练。

## 5. 实验分析

作者通过23人正式实验、模型间统计比较和TranSalNet修复实验验证方法。代表性结论是：人物和手部图像最难做到无缺陷；Imagen无缺陷数量较多但更易出现CG感和全局风格问题，Midjourney较真实却有较多局部细节缺陷。TranSalNet能产生更接近人类判断的缺陷定位，缺陷引导修复可改善结果。

优势是标注粒度高、覆盖构图关系、可直接服务下游任务。局限是数据规模较小、每个提示词只由一个模型生成一张图，模型分配可能造成混杂；提示词和标签依赖人工，主观性较强，且训练超参数、划分策略和统计显著性描述不足。

## 6. 实用指南

数据和补充材料已在GitHub开源：`Future-IQA/CO-AID`。复现需下载参考图，重建人工构图提示词，调用三种T2I模型，按全局/局部层级设计标注界面，再将点击点转为热图微调TranSalNet。需特别控制模型版本、生成参数、随机种子、图像尺寸和参与者一致性。该框架可迁移到视频帧、3D渲染、广告设计等任务，只需重新定义实体类别、缺陷标签和空间标注形式。

## 7. 总结

**核心思想：**用人类定位复杂构图缺陷，反向优化AI生成。

**速记版Pipeline：**

1. 收集复杂真实构图图像。  
2. 生成并人工强化多实体提示词。  
3. 用多个T2I模型生成样本。  
4. 人类标注缺陷位置与原因。  
5. 学习缺陷热图并引导图像修复。

**Key Findings:**

- State-of-the-art text-to-image (T2I) models exhibit pronounced and systematic defects when prompts involve intricate compositional factors such as multiple entities and multiple attributes.
- Experimental results show that training a deep model on CO-AID can both predict defects in AI-generated images and optimize AI image generation, demonstrating its usability and effectiveness.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.25933v1)
- [arXiv](https://arxiv.org/abs/2608.25933v1)

---

<a id='2608.25924v1'></a>
## [Visual General Intelligence: A White Paper](https://arxiv.org/abs/2608.25924v1)

**Authors:** Hirokatsu Kataoka, Yoshihiro Fukuhara, Yonglong Tian, Shangzhe Wu, Oishi Deb, Ryousuke Yamada, Christian Rupprecht, Jianyuan Wang, Kohsuke Ide, Koichi Namekata, Xianzheng Ma, Yiming Chen, Robert Geirhos, Aditi Raghunathan, Yuki M. Asano, Deva Ramanan, David Fouhey, Andrew J. Davison, Yilun Du, Jiajun Wu, Zhuang Liu

**Published:** 2026-08-26

**Categories:** cs.CV

**Abstract:**

This paper reconsiders intelligence from a vision-centered perspective and examines whether intelligence emerging from visual experience and learning may provide a pathway toward AGI. In the language domain, beginning with the introduction of the Transformer architecture, the GPT series has demonstrated transfer to unseen tasks through autoregressive language modeling on web-scale text combined with aggressive scaling. This raises a natural question, namely, what capabilities and forms of intelligence can emerge from visual modalities such as images, videos, and geometry? In this paper, we discuss whether visual intelligence can serve as a pathway toward AGI, referred to in this paper as visual general intelligence (VGI), by bringing together contributors from diverse standpoints and affiliations. Our aim is not to offer a single definition of visual intelligence, but to clarify the principles that computer vision should pursue in the AGI era, the visual input modalities, the benchmarks, the learning paradigms, and the relationship between vision, when taken as the core, and other modalities such as language.

**Analysis:**

## 1. 摘要翻译

本文从视觉中心的视角重新思考智能，探讨智能是否能够从视觉经验与学习中产生，并进一步成为通向通用人工智能（AGI）的路径，称为视觉通用智能（VGI）。语言领域中，Transformer 与 GPT 表明，在海量文本上进行自回归建模并持续扩展规模，能够迁移到未见任务并产生上下文学习等能力。本文据此追问：图像、视频和几何等视觉模态能够产生哪些能力与智能形态？文章不试图给出唯一的 VGI 定义，而是综合不同观点，讨论视觉研究在 AGI 时代应追求的原则、视觉输入形式、评测基准、学习范式，以及视觉与语言等其他模态的关系。

## 2. 方法动机分析

**驱动力：**本文认为，视觉不应只是语言模型的输入接口，而可能是独立产生世界知识与智能的基础。视觉包含空间、几何、运动、物理和因果变化等语言未充分描述的信息。

**现有痛点：**当前视觉系统通常是任务专用、训练后冻结、依赖大量标注或人工整理数据；生成模型即使能产生逼真视频，也未必真正掌握质量、摩擦、接触、关节等物理结构；静态 benchmark 也无法衡量持续学习、主动观察、创造性和真实行动能力。

**核心假设：**若模型能从连续视觉经验中学习“世界如何存在、如何变化以及如何被行动改变”，并结合记忆、生成、重建和交互，就可能从视觉功能走向视觉智能。

## 3. 方法设计详解

本文不是提出一个已实现的单一模型，而是提出一个综合性研究框架。其潜在 pipeline 为：

1. **输入经验：**接收图像、视频、3D/4D观测，并可联合音频、触觉、本体感觉等信号；数据应具有多样性，而非简单扩大重复样本。
2. **结构化感知：**从视觉流中发现对象、部件、表面、运动、相机位姿和遮挡关系，形成对象级或场景级表示。
3. **世界建模：**维护持久、可更新的空间记忆，表示实体身份、几何、材料、姿态、关系、动力学及不确定性。其思想接近 SLAM，但进一步加入语义、物理与任务相关信息。
4. **三类学习目标协同：**  
   - 顺序预测：预测下一视觉状态，迫使模型学习运动、因果和长期变化；  
   - 开放生成：生成图像、视频、3D场景及反事实未来，支持想象与规划；  
   - 重建：从遮挡、缺失或多视角观测中恢复深度、结构和隐藏原因。
5. **主动交互：**系统选择“看哪里、看多久、需要多少细节”，通过移动视角、推拉物体等行动消除歧义，并用预测结果与真实反馈校正模型。
6. **持续学习与行动：**将新经验写入多时间尺度记忆，避免灾难性遗忘；高层视觉生成器提出任务轨迹，低层控制器执行动作，执行误差反过来更新世界模型。

在表示层面，论文主张连续嵌入与离散结构结合：程序或符号描述对象层级、关系和约束，神经表示承载难以命名的精细外观与几何。

## 4. 方法对比分析

**本质区别：**主流视觉模型多针对识别、分割或生成结果优化；VGI 关注可迁移、可验证、可编辑、可持续更新的世界知识。它也区别于“视觉编码器+LLM”：视觉应先形成独立的空间与物理理解，而不是仅服务于语言推理。

**创新贡献：**贡献主要是提出新的问题定义与评价维度，而非新网络结构：将生成、预测、重建、空间记忆、主动感知、具身行动和持续学习统一为 VGI 的候选组成。

**适用场景：**机器人、自动驾驶、科学发现、长期环境监测、3D内容生成和需要反事实推理的交互系统。

## 5. 实验分析

本文为白皮书，没有统一模型、数据集、消融实验或定量结果。文中引用的生成视频零样本任务、3D记忆、机器人自改进等工作只能作为方向性证据，不能视为本文实验验证。

**优势：**视角全面，明确指出“逼真生成≠物理理解”，并把长期记忆和主动交互纳入智能定义。  
**局限：**缺少可执行算法、统一 benchmark 和成功判据；各观点之间尚未解决规模、结构、效率与持续学习的权衡。

## 6. 实用指南

论文自身未提供可复现代码或完整实现。复现其思想可从视频生成/预测模型出发，加入3D场景记忆、对象跟踪、动作条件生成和在线更新机制，并用跨任务迁移、长期一致性、物理干预、主动观测和能耗进行评测。迁移到机器人时，应增加本体感觉与触觉；迁移到科学任务时，应加入仪器系统误差、物理单位及领域知识约束。

## 7. 总结

**核心思想：**从视觉经验中学习可行动的世界模型。

**速记版 pipeline：**

1. 连续观察多样视觉世界；  
2. 发现并记住对象、空间和变化规律；  
3. 预测未来、补全隐藏结构、生成反事实；  
4. 主动选择观察和行动以验证预测；  
5. 根据反馈持续更新并迁移到新任务。

**Key Findings:**

- In this paper, we discuss whether visual intelligence can serve as a pathway toward AGI, referred to in this paper as visual general intelligence (VGI), by bringing together contributors from diverse standpoints and affiliations.
- Our aim is not to offer a single definition of visual intelligence, but to clarify the principles that computer vision should pursue in the AGI era, the visual input modalities, the benchmarks, the learning paradigms, and the relationship between vision, when taken as the core, and other modalities such as language.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.25924v1)
- [arXiv](https://arxiv.org/abs/2608.25924v1)

---

<a id='2608.25917v1'></a>
## [Choose Your Game Wisely: Measuring Game-Theoretic Structures in Real-World Vehicle Interactions](https://arxiv.org/abs/2608.25917v1)

**Authors:** Yueyuan Li, Rongcheng Nie, Weijie Xi, Mingyang Jiang, Songan Zhang, Hanyang Zhuang, Ming Yang

**Published:** 2026-08-26

**Categories:** cs.AI, cs.RO

**Abstract:**

Game-theoretic models provide principled frameworks for modeling vehicle interactions, but their underlying temporal assumptions have not been systematically examined against real-world driving behavior. In particular, it remains unclear how simultaneous, sequential, and asymmetric interaction structures can be measured from vehicle trajectories. This paper develops a trajectory-based interaction measurement framework to identify interaction events and quantify behavioral change onset, temporal organization, post-onset response dynamics, and ordering stability. The framework uses behavioral deviations to verify candidate interactions. We evaluate the framework on six real-world trajectory datasets, including INTERACTION, highD, inD, rounD, Waymo Open Motion, and nuPlan, covering diverse road geometries, traffic environments, and interaction types. The results show that concurrent and sequential behavioral changes both constitute substantial proportions of observed following, merging, and conflicting interactions. Among sequential interactions, stable ordering is more prevalent than alternating ordering, indicating that persistent asymmetric roles are a common interaction structure. Importantly, temporal precedence does not necessarily coincide with a measurable behavioral response, indicating that temporal ordering alone may not be sufficient to characterize behavioral dependence. These findings show that real-world interactions exhibit concurrent, sequential, and persistently ordered temporal structures. Different game-theoretic formulations are therefore better regarded as complementary modeling abstractions for different interaction regimes rather than as a universal structure governing all vehicle interactions.

**Analysis:**

# 论文方法分析与总结

## 1. 摘要翻译

博弈论模型为刻画车辆交互提供了系统框架，但其潜在的时间假设尚未被系统地与真实驾驶行为进行对照。尤其是，同时、顺序和非对称交互结构如何从车辆轨迹中测量，仍不清楚。本文提出一种基于轨迹的交互测量框架，用于识别交互事件，并量化行为变化起点、时间组织形式、起点后的响应动态以及顺序稳定性。该框架利用行为偏差验证候选交互。作者在 INTERACTION、highD、inD、rounD、Waymo Open Motion 和 nuPlan 六个真实轨迹数据集上进行评估。结果表明，同时和顺序行为变化在跟驰、合流和冲突交互中均占有较大比例；在顺序交互中，稳定顺序明显多于交替顺序，说明持续的非对称角色较为常见。重要的是，时间上的先后并不必然伴随可测量的行为响应，因此仅凭时间顺序不足以刻画行为依赖。真实车辆交互同时具有并发、顺序和持续有序等结构，不同博弈论形式应被视为适用于不同交互状态的互补建模抽象，而非所有交互都遵循的统一结构。

## 2. 方法动机分析

**驱动力与痛点：**现有研究通常利用轨迹判断“是否交互”、相关性强弱或预测关系，但交互的时间组织方式往往由模型预先指定：Nash通常对应同时决策，Stackelberg对应领导者—跟随者的顺序决策。因此，已有方法更多是用数据拟合既定博弈结构，而不是从真实行为中测量该结构。

**核心假设：**如果车辆确实受到交互影响，其运动应相对于道路几何约束下的正常参考行为产生显著、持续的偏差；通过比较两车行为变化的时间关系，可观察地近似同时、顺序和非对称角色结构。但作者明确不把时间先后等同于因果响应。

## 3. 方法设计详解

### Pipeline

1. **候选交互提取**  
   为每辆车构造参考轨迹：保持当前速度，沿车道路线驶向观测终点，并转换为考虑车辆占用的运动走廊。依据道路拓扑、空间兼容性和时间兼容性提取三类关系：同车道同向跟驰、不同车道同向合流、未来路径相交或汇聚的冲突。逐帧检测再聚合为事件窗口。

2. **行为证据筛选**  
   仅有空间接近不代表交互。对事件窗口 \(W\)，构造“窗口起始速度恒定”的参考速度 \(v_i^w(t)\)，计算观测速度与参考速度差的中位绝对偏差，并用车辆自身非交互控制窗口的均值 \(\mu_i\) 和标准差 \(\sigma_i\) 标准化：
   \[
   z_i=\frac{\operatorname{median}|v_i^{obs}-v_i^w|-\mu_i}{\sigma_i+0.5}.
   \]
   当 \(\max(z_i,z_j)>2\) 时保留事件，因此即使只有一辆车出现明显调整，也能保留非对称交互。合流和冲突还要求两车到达共享区域的时间差不超过1秒。

3. **行为变化起点检测**  
   相对于考虑曲率、停车线等道路约束的地图参考速度，定义速度残差：
   \[
   r_i(t)=v_i^{obs}(t)-v_i^{ref}(t).
   \]
   当 \(|r_i|\) 持续超过0.3 m/s至少400 ms时，将首次满足条件的时刻定义为行为变化起点。该设计通过持续时间抑制轨迹噪声和瞬时抖动。

4. **时间组织判定**  
   比较两车起点差 \(\Delta t\)。两车均检测到起点且差值不超过400 ms，判为并发；否则判为顺序，并根据符号确定谁先变化；只有一车有起点为单侧，无起点为未解析。进一步，作者不只比较首次变化，而是分析整个变化序列：少于两次顺序反转为稳定顺序，至少两次反转为交替顺序。

5. **响应动态分析**  
   对“先变化车辆→后变化车辆”的有向关系，利用源车变化前目标车的残差均值和方差构造自适应阈值：
   \[
   \theta_j=\max(0.3,\mu_{j,pre}+2\sigma_{j,pre}).
   \]
   若目标车在源车起点后2秒内出现持续超阈值变化，则判为响应并计算响应延迟；若目标车此前已变化，则为先行动；否则为无响应或右删失。由此将“时间先后”与“可观测行为依赖”分离。

6. **统计汇总**  
   使用场景级聚类Bootstrap（2000次）估计置信区间，并用带场景随机截距的逻辑混合模型，控制事件时长和车辆数，比较不同交互类型的并发概率。

## 4. 方法对比与创新

其本质区别不是提出新的车辆决策模型，而是把Nash、顺序博弈和Stackelberg的时间含义转化为可从轨迹测量的行为指标。创新主要包括：  
①用个体化行为基线验证交互，避免把几何接近误判为交互；  
②区分首次起点、完整顺序、响应延迟和顺序稳定性；  
③揭示“先发生”不等于“引起响应”，为博弈结构选择提供经验依据。适合做真实驾驶行为分析、交互数据标注和博弈模型选择，不适合直接推断驾驶者的意图、信息集、效用函数或因果机制。

## 5. 实验分析

作者在六个数据集、跟驰/合流/冲突场景上验证框架。代表性结论是：并发和顺序事件均大量存在，比例约为34%—59%；顺序事件中稳定顺序占80.46%—100%，显著高于交替顺序。另有约41%的可评估有向关系出现明确响应，说明时间先后不能直接替代行为依赖。

**优势：**跨数据集、无需训练、具有个体基线、能分解多层时间结构。  
**局限：**主要依赖速度偏差，可能遗漏转向、加速度、间距或非运动信号；阈值和参考轨迹具有经验性；测量结果仍是行为相关而非因果证明。

## 6. 实用指南

文中未说明提供完整开源实现，复现需自行完成数据适配、地图参考轨迹和占用走廊构造。关键参数为：行为偏差阈值2.0、残差阈值0.3 m/s、持续时间400 ms、并发容差400 ms、共享区域时间差1 s、响应窗口2 s。迁移到其他任务时，可将速度残差替换或扩展为加速度、横向偏移、车头时距等多模态行为残差，并重新建立个体化非交互基线。

## 7. 总结

**核心思想：**从轨迹行为变化测量真实交互结构。

**速记版Pipeline：**
1. 用道路和时间关系找候选车辆对；  
2. 用车辆自身正常行为基线筛掉“仅接近不交互”事件；  
3. 检测两车持续行为变化的起点；  
4. 比较变化先后及全过程顺序是否稳定；  
5. 检查后车是否真正出现起点后的新响应。

**Key Findings:**

- The results show that concurrent and sequential behavioral changes both constitute substantial proportions of observed following, merging, and conflicting interactions.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.25917v1)
- [arXiv](https://arxiv.org/abs/2608.25917v1)

---

<a id='2608.25872v1'></a>
## [VISTA: Visually Inferred Spatial ConTact Attention for Contact-Rich Manipulation](https://arxiv.org/abs/2608.25872v1)

**Authors:** Jiayi Chen, Wenlong Dong, Yan Huang, Xianglin Chen, Zijian Lin, Jiaqi Yin, Yushan Liu, Wenbo Ding

**Published:** 2026-08-26

**Categories:** cs.RO

**Abstract:**

Contact-rich manipulation requires precise interaction feedback. While vision-centric imitation learning is prevalent, external visual observations provide indirect and ambiguous cues about contact states, particularly under occlusion or subtle object--gripper interactions; dedicated tactile or force sensors can provide rich contact information but introduce additional hardware complexity, calibration requirements, and deployment costs. To bridge this gap, we propose VISTA-Policy, an imitation learning paradigm that utilizes the Visual Deformation Field (VDF), a 3D displacement representation of a compliant gripper, as high-dimensional visuo-physical feedback. The framework integrates: 1) a Physics-Aware Encoding Engine for real-time VDF decoding; 2) an Energy Aggregation Denoising Mechanism to isolate true interaction signals; and 3) a Deformation-Augmented Policy Network with incremental gripper actions for precise closed-loop correction. Extensive evaluations on Cross-Scale Object Grasping, Cap Unscrewing, and Calligraphy Writing demonstrate that VISTA-Policy outperforms the strong pure-vision baseline 3D Diffusion Policy and the tactile baseline. VISTA-Policy further demonstrates substantial out-of-distribution generalization to unseen object scales and robustness against dynamic disturbances, offering a durable and cost-effective route toward general-purpose fine-grained manipulation in unstructured environments. Project videos and supplementary materials are available at: https://sites.google.com/view/vista-policy.

**Analysis:**

# 论文方法分析与总结

## 1. 摘要翻译

接触丰富型操作需要精确的交互反馈。尽管视觉模仿学习已广泛应用，但外部视觉通常只能间接、模糊地反映接触状态，尤其在接触区域被遮挡或物体与夹爪之间发生细微交互时更为明显。触觉和力传感器能够提供丰富的接触信息，却带来额外硬件、标定和部署成本。

为此，本文提出 **VISTA-Policy**，利用柔顺夹爪的三维位移表示——**视觉形变场（Visual Deformation Field, VDF）**——作为高维视觉-物理反馈。方法包含：实时解码VDF的物理感知编码引擎；用于隔离真实交互信号的能量聚合去噪机制；以及采用增量式夹爪动作的形变增强策略网络。实验表明，在跨尺度抓取、瓶盖旋拧和书法书写任务中，VISTA-Policy优于纯视觉DP3和触觉基线，并表现出较强的尺度外推与动态扰动鲁棒性。

## 2. 方法动机分析

**驱动力：**传统视觉观察主要描述物体外观、几何和机器人位姿，却没有直接呈现“是否接触、接触多强、是否滑脱”等物理状态。作者的关键视角是：柔顺夹爪的形变正是接触造成的可见物理响应，因此可作为低成本触觉替代物。

**现有痛点：**点云难以识别细微或遮挡接触；绝对夹爪开度使策略容易记住训练物体尺寸，难以处理未见尺度；触觉传感器存在磨损、成本、平面接触失效及跨设备不一致等问题。

**研究假设：**不同物体在相似局部接触状态下，会诱发相似的夹爪形变；若策略直接依据这种形变进行闭环控制，就能学习更具物理不变性的操作规律。

## 3. 方法设计详解

### Pipeline

1. **在线夹爪分割与跟踪**  
输入RGB-D图像。作者将Track-Anything改造成状态机式流式推理：当前帧结合上一时刻隐式记忆，仅执行单步更新，使时间复杂度由序列式的 \(O(T)\) 降为 \(O(1)\)，系统约以20 Hz运行。网络重点提取夹爪前端内侧边缘，以提高遮挡和光照变化下的稳定性。

2. **边缘重建与三维VDF提取**  
对分割结果进行形态学开运算、面积过滤，并分别用B样条拟合左右指尖边缘，修复断裂和噪声。利用RGB-D将边缘像素反投影到三维空间，以夹爪刚性根部建立局部坐标系，并沿纵向 \(y\) 轴等距采样 \(N\) 个位置。每个采样点记录左右边缘相对刚性根部的位移：
\[
d_{i,t}=[\Delta x_L,\Delta z_L,\Delta x_R,\Delta z_R]\in\mathbb{R}^4.
\]
因此VDF为 \(D\in\mathbb{R}^{T\times N\times4}\)。使用相对根部的位移而非绝对位置，可削弱夹爪整体移动与接触状态之间的伪相关；缺失深度则沿特征列线性插值。

3. **空间能量去噪与接触门控**  
单个点的异常位移可能造成误触发，因此先设置噪声底 \(\tau_{\text{noise}}\)，计算所有采样点超过噪声底后的平均形变能量：
\[
E_t=\frac1N\sum_i\mathrm{ReLU}(\|d_{i,t}\|_2-\tau_{\text{noise}}).
\]
再通过Sigmoid将其映射为接触置信度 \(c^{raw}_t\)，并用EMA平滑：
\[
c_t=\alpha c^{raw}_t+(1-\alpha)c_{t-1}.
\]
最终采用软门控：
\[
\tilde f_{\text{def}}=c_t f_{\text{def}}+(1-c_t)e_{\text{neutral}}.
\]
也就是说，未接触时不直接把噪声送入策略，而是替换为可学习的“中性状态”。

4. **形变增强策略与动作输出**  
视觉场景特征、机器人状态和门控后的VDF特征拼接为多模态状态。VDF先由1D CNN沿夹爪采样拓扑提取局部空间模式，再结合最大池化和平均池化：前者捕捉局部峰值形变，后者描述整体接触分布。随后，时间序列输入Transformer，并通过可学习readout token聚合整个操作窗口的接触动态。

控制上，夹爪不再预测绝对开度，而是预测相邻时刻的开度增量，并叠加到当前实际开度。这样策略可持续依据实时形变调整夹爪，而不受训练集中绝对尺寸边界限制。

## 4. 方法对比分析

**本质区别：**DP3从点云推断几何并生成动作；VISTA额外引入由接触直接诱发的“局部物理响应”，并将其作为门控后的闭环信号。相比触觉方法，VISTA不在夹爪内部安装传感器，而是通过外部相机观察被动柔顺结构。

**创新点：**  
1. 将柔顺夹爪形变构造成结构化三维VDF；  
2. 用空间能量聚合替代单点或单纯时间滤波，抑制非接触噪声；  
3. 将VDF与增量夹爪动作联合设计，提升跨尺度外推；  
4. 采用空间CNN—时间Transformer建模形变的时空演化。

适合抓取、旋拧、插拔、工具使用及需要柔顺接触的任务，前提是夹爪形变可被外部相机稳定观测。

## 5. 实验分析

作者在跨尺度抓取、瓶盖旋拧和书法书写上，与DP3、带腕部点云的DP3及Daimon视觉触觉传感器比较，并进行动作空间、去噪和编码器消融。最具代表性的结果是：VISTA在三类主任务中分别达到100%、90%、100%的成功率；在多尺度抓取中仅用20条示范即可实现训练及未见尺度的100%成功率。消融表明，去噪门控和时空编码均不可缺少。

主要优势是低成本、物理反馈明确、样本效率高、具备尺度泛化和扰动恢复能力。局限是严重遮挡或低照度下VDF提取会退化；当前VDF主要表示接触状态，尚未建立形变与真实力、材料柔顺性的定量映射。

## 6. 实用指南

论文仅明确提供项目视频和补充材料链接，未说明完整代码开源。复现需准备UR3、UMI被动柔顺夹爪、L515/D405 RGB-D相机，并完成相机标定、夹爪根部坐标系标定、噪声底与EMA参数校准。关键实现包括稳定的边缘分割、B样条修复、深度补全、VDF时空窗口构造，以及增量夹爪动作标签。迁移到其他任务时，应重新定义可观测形变区域、局部坐标系和接触能量阈值；若换用不同夹爪，需重新建立其形变与接触状态的对应关系。

## 7. 总结

**核心思想：**用夹爪可见形变替代触觉反馈。

**速记版Pipeline：**

1. 相机实时分割并跟踪柔顺夹爪边缘；  
2. 将边缘变化转换为相对根部的三维形变场；  
3. 聚合全局形变能量，判断并平滑接触状态；  
4. 用CNN和Transformer理解形变的空间与时间变化；  
5. 根据视觉、机器人状态和形变反馈，输出夹爪增量动作。

**Key Findings:**

- To bridge this gap, we propose VISTA-Policy, an imitation learning paradigm that utilizes the Visual Deformation Field (VDF), a 3D displacement representation of a compliant gripper, as high-dimensional visuo-physical feedback.
- Extensive evaluations on Cross-Scale Object Grasping, Cap Unscrewing, and Calligraphy Writing demonstrate that VISTA-Policy outperforms the strong pure-vision baseline 3D Diffusion Policy and the tactile baseline.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.25872v1)
- [arXiv](https://arxiv.org/abs/2608.25872v1)

---

<a id='2608.25864v1'></a>
## [MA-VLA: Multi-Arm Vision-Language-Action Model for Collaboration and Compositional Generalization](https://arxiv.org/abs/2608.25864v1)

**Authors:** Zaibin Zhang, Junlan Xiao, Zhongbo Zhang, Yifan Wang, Li Kang, Yiran Qin, Changxing Xia, Heng Zhou, Talas Fu, Enshen Zhou, Ruimao Zhang, Zhenfei Yin, Huchuan Lu, Lijun Wang

**Published:** 2026-08-26

**Categories:** cs.RO

**Abstract:**

Multi-arm collaboration is becoming a core capability in embodied manipulation. Recent vision-language-action (VLA) models integrate perception, language, and control, but most represent language as a single global instruction and do not provide an explicit mechanism for assigning and composing arm-specific behaviors. This design limits transfer to collaboration patterns that differ from those observed during training. We present MA-VLA, a unified framework for multi-arm collaboration via atomic action assignment. MA-VLA decomposes cooperative behavior into mid-level atomic prompts and allocates them to individual arms, enabling explicit subgoal specification and compositional reuse across tasks. To reduce reliance on fixed execution roles, we introduce Arm Shuffle, a training-time permutation of the observation, state, and assigned atomic prompts for each arm. This permutation enforces role-agnostic instruction following and supports recomposition into unseen coordination patterns, which we term multi-arm compositional generalization. We also construct a benchmark in which test-time collaboration patterns are absent in training set. Across simulation and real-world evaluations, prior state-of-the-art VLAs largely fail under these unseen collaborations, while MA-VLA consistently succeeds. These results indicate that structured, per-arm atomic action assignment offers a practical route to scalable generalization in multi-arm embodied systems. Code, models, and data are available at https://github.com/zhangzaibin/future-robots

**Analysis:**

## 1. 摘要翻译

多臂协作正成为具身操作的核心能力。近期视觉-语言-动作（VLA）模型融合了感知、语言与控制，但大多将语言表示为单一全局指令，缺乏显式分配和组合各机械臂行为的机制，因此难以迁移到训练中未出现的协作模式。本文提出 MA-VLA，通过“原子动作分配”实现多臂协作：将协作行为分解为中层原子提示，并分配给不同机械臂，从而明确指定子目标并支持跨任务组合复用。为减少模型对固定执行角色的依赖，作者提出 Arm Shuffle，在训练时同步打乱各机械臂的观测、状态及原子提示，促使模型学习与机械臂身份无关的指令执行，并重组出未见过的协作模式。实验表明，现有VLA在未见协作结构下大多失效，而MA-VLA在仿真和真实平台上均表现稳定。

## 2. 方法动机分析

**驱动力：** 多臂任务的难点不是单臂“会不会抓取”，而是“谁在何时执行什么动作”。作者希望把隐式学习的分工结构显式化，以提升组合泛化能力。

**现有痛点：** 主流VLA通常将全局指令直接映射为联合动作，机械臂角色、空间位置和协作顺序被模型隐式记忆，容易形成固定的“左臂/右臂”偏置；独立训练多个单臂策略又缺乏跨臂协调。

**核心假设：** 若将任务拆成可复用的原子动作，并训练模型理解“原子动作—机械臂”的动态分配关系，则未见过的协作任务可通过重新组合已学动作完成。

## 3. 方法设计详解

### Pipeline

1. **任务分解。** 输入高层指令、场景图像和预定义原子提示集合，例如“抓取物体、保持、对齐、放置”。VLM规划器将任务生成阶段序列  
   \[
   \mathbf p_t=(p_t^1,\ldots,p_t^N)
   \]
   每个阶段为每条机械臂指定一个原子子指令。规划器采用GPT-4.1，输出被规范化到有限模板集合中；它只负责语言规划，不直接输出控制量。

2. **统一语言编码。** 将同一时刻所有机械臂的提示拼接为“Arm 1: …, Arm 2: …”的联合指令，使执行器同时获得全局协作意图，而非各臂独立决策。

3. **多模态执行。** 执行器输入全局视图、各臂腕部视图、所有机械臂的本体状态及联合提示，在一次前向传播中输出全部机械臂动作。共享视觉—语言特征用于建模臂间依赖，多头投影层再将共享表示拆分为各臂动作头。

4. **连续动作生成。** 基于pi0的流匹配专家，从噪声动作潜变量逐步积分到真实动作，产生平滑、物理一致的动作轨迹，并以行为克隆损失监督各臂动作。

5. **角色扰动训练。** Arm Shuffle以概率 \(p_{\text{shuffle}}\) 同步置换机械臂的状态、局部视图、原子提示和动作标签，保持四者对应关系不变，防止模型把某种行为绑定到固定编号或空间位置。View Dropout则随机屏蔽部分相机，迫使模型利用冗余视角。训练目标仍是各臂动作损失之和。

### 关键设计价值

MA-VLA不是简单增加一个任务标签，而是将语言变成“协作接口”：规划器负责显式分工，统一执行器负责联合控制，Arm Shuffle负责消除身份依赖。三者共同实现“已知动作的未知组合”。

## 4. 方法对比与适用性

**本质区别：** 传统VLA是“全局指令→联合动作”；MA-VLA是“全局指令→分阶段、逐臂原子提示→联合动作”。相比独立单臂模型，它保留了共享上下文和跨臂协调能力。

**创新点：**  
1. 原子动作分配作为多臂协作的中层表示；  
2. Arm Shuffle实现角色无关的指令跟随；  
3. 将组合泛化明确形式化为训练外的角色、顺序和交互结构重组。

适用于双臂至多臂协作、堆叠、传递、流水线操作等角色可交换任务；对强实时规划、复杂长程因果依赖和原子动作集合之外的新技能，适用性有限。

## 5. 实验分析

作者在RoboFactory、RoboTwin 2.0及SO101真实双臂平台上，与DP、ACT、Pi0等比较，并设置未见协作顺序和角色分配的OOD测试。代表性结论是：OOD任务中基线成功率普遍为零，MA-VLA平均达到13%；真实平台中，Pi0的OOD任务全部失败，而MA-VLA在四类任务上取得非零成功。消融实验显示，Arm Shuffle是泛化提升的主要来源，View Dropout进一步增强鲁棒性。

**优势：** 分工可解释、单模型联合推理、对角色重排更稳健。  
**局限：** 规划器依赖外部GPT-4.1，成本、延迟和可复现性受限；原子提示和终止条件需要人工设计；OOD成功率仍较低，且对更复杂协作关系的验证不足。

## 6. 实用指南

论文公开代码、模型和数据。复现需准备原子动作标注、同步多视角与本体状态，使用pi0初始化；仿真训练约1万至3万步，batch size为32，动作horizon为50。组合泛化任务应启用Arm Shuffle和View Dropout，文中常用 \(p_{\text{shuffle}}=0.6\sim1.0\)、\(p_{\text{drop}}=0.2\sim0.4\)。迁移到其他任务时，主要改造原子提示词表、状态判据和数据解析器，并保持各臂输入—提示—动作同步置换。

## 7. 总结

**核心思想：** 原子分工驱动多臂组合泛化。

**速记版Pipeline：**  
1. 把总任务拆成分阶段的小动作。  
2. 给每个小动作指定执行机械臂。  
3. 将所有机械臂信息交给一个联合策略。  
4. 训练时随机交换机械臂角色并遮挡部分视角。  
5. 用已学动作重新组合未见协作任务。

**Key Findings:**

- We present MA-VLA, a unified framework for multi-arm collaboration via atomic action assignment.
- To reduce reliance on fixed execution roles, we introduce Arm Shuffle, a training-time permutation of the observation, state, and assigned atomic prompts for each arm.
- Across simulation and real-world evaluations, prior state-of-the-art VLAs largely fail under these unseen collaborations, while MA-VLA consistently succeeds.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.25864v1)
- [arXiv](https://arxiv.org/abs/2608.25864v1)

---

<a id='2608.25851v1'></a>
## [DEFUSE: Generalizable Backdoor Defense for Self-Supervised Encoders with Generative Priors](https://arxiv.org/abs/2608.25851v1)

**Authors:** Tuo Chen, Jie Gui, Minjing Dong, Lanting Fang, Ju Jia, Benlei Cui, Jian Liu

**Published:** 2026-08-26

**Categories:** cs.CV

**Abstract:**

Self-supervised learning (SSL) encoders are vulnerable to backdoor attacks, posing threats to both visual SSL encoders and vision-language encoders. Existing defenses are typically designed for only one of these paradigms and rely on restrictive assumptions such as access to uninfected in-distribution data or precomputed pseudo-labels, which are difficult to satisfy in practice. To address these limitations, we propose DEFUSE, a generalizable backdoor detection framework for SSL encoders. Inspired by Bayesian posterior inference, we reformulate backdoor detection as a representation-conditioned image likelihood estimation problem parameterized by a conditional diffusion generative model. Uninfected representations tend to yield semantically consistent reconstructions, whereas backdoored ones are more likely to be mapped to the attacker's target class or semantically meaningless images, deviating from the original semantics and thereby exposing the backdoor. However, we find that the exact likelihood is intractable, because highly abstracted representations discard the low-level information necessary for pixel-faithful reconstruction. We therefore relax the objective to semantic reconstruction and evaluate it in a well-separated representation space provided by a reference encoder. Rather than training from scratch, we fine-tune a pretrained diffusion model, leveraging its generative prior to map data onto the natural image manifold while preserving semantic content. Extensive experiments demonstrate that DEFUSE substantially outperforms existing detectors across diverse attack settings, generalizing to both visual SSL and vision-language encoders. Notably, our method greatly reduces the reliance on prior knowledge about the victim encoder or the attack strategy. The source code is available at https://github.com/jsrdcht/DEFUSE .

**Analysis:**

# 1. 摘要翻译
自监督学习（SSL）编码器容易受到后门攻击，既威胁视觉自监督编码器，也威胁视觉语言编码器。现有防御通常只适用于某一种范式，并依赖干净的分布内数据或预计算伪标签等实践中难以获得的信息。为此，本文提出 DEFUSE，一种具有普适性的 SSL 编码器后门检测框架。受贝叶斯后验推断启发，作者将后门检测重构为“由表示条件化的图像似然估计”问题，并使用条件扩散模型实现。干净表示通常能够产生与原图语义一致的重建，而后门表示更可能生成攻击目标类别或语义无关图像，因此会偏离原始语义。由于高度抽象的表示缺失像素级重建所需的低层信息，精确似然难以计算，作者进一步将目标放宽为语义重建，并利用参考编码器在区分性较强的表示空间中评价一致性。方法不从零训练生成模型，而是微调预训练扩散模型，借助其生成先验将结果约束到自然图像流形。实验表明，DEFUSE 在多种攻击下明显优于现有检测器，并可同时适用于视觉 SSL 与视觉语言编码器。

# 2. 方法动机
**驱动力：**后门输入的像素可能仍然正常，但其经编码器映射后的语义表示被强行推向攻击者目标。因此，检测重点不应是寻找固定触发器，而应判断“表示是否仍能解释原图语义”。

**现有痛点：**已有方法往往绑定 CLIP 或某一类视觉 SSL；依赖干净数据、伪标签、可见色块等先验；像素重建误差还会把视角、布局和纹理复杂度误判为异常。

**核心假设：**干净表示保留了原图的主要语义，后门表示与原图语义错配；若将表示还原成图像，后门样本的重建与原图在可靠语义空间中的相似度会显著降低。

# 3. 方法设计详解
**Pipeline：**

1. 输入图像 \(x\)，通过待检测编码器得到 \(z=f(x)\)。编码器仅需黑盒查询，不需要知道训练过程或攻击类型。  
2. 用公开图像集计算 \((x_i,z_i)\)，将 \(z_i\) 经线性投影 \(W\) 转成4个扩散模型条件 token。  
3. 将这些 token 注入预训练 SDXL 等条件扩散模型的 cross-attention。训练时冻结大部分生成先验，仅更新条件相关模块和 \(W\)，以噪声预测损失学习 \(z\rightarrow x\) 的语义映射。  
4. 推理时固定随机种子或采用 DDIM，从噪声逐步生成重建图 \(\hat{x}\)。生成先验负责补充表示中丢失的低层细节，同时限制输出落在自然图像流形。  
5. 使用独立参考编码器 \(\phi\)（默认 DINOv2）计算  
\[
s(x)=\cos(\phi(x),\phi(\hat{x})).
\]
分数低于阈值 \(\tau\) 判为后门。这里并非直接计算严格的 \(P(x|z)\)，而是用语义一致性作为其可行替代。

**结构协同：**待测编码器提供可能被攻击的语义条件；扩散模型提供自然图像先验；参考编码器提供与生成模型解耦的语义判别空间。三者共同把“隐藏空间异常”转化为“原图—重建图语义不一致”。

# 4. 对比与创新
DEFUSE 不逆向搜索触发器、不聚类伪标签，也不依赖攻击类别，而是检测表示到图像的可解释性。其主要创新是：  
1. 首次将 SSL 后门检测系统化为表示条件图像生成问题；  
2. 用预训练扩散先验解决全局表示导致的欠定重建；  
3. 将检测指标从脆弱的像素距离改为参考语义空间相似度。  
适合模型供应链审计、第三方编码器上线前检测，以及同时包含视觉 SSL 和 CLIP 类模型的场景。

# 5. 实验分析
作者在 CLIP、SimSiam-ResNet18 上测试7类数据投毒和模型操纵攻击，并与 DECREE、DBCL、DEDE、PatchSearch 比较。DEFUSE 在主实验中各攻击 AUPRC 约为0.81–0.96；对抗性优化使 AUROC 从0.78下降后，加高斯噪声可恢复至0.90。  
**优势：**跨范式、少攻击先验、对部分投毒较稳健。  
**局限：**需要额外微调扩散模型，推理成本较高；性能依赖生成质量、参考编码器和阈值校准；白盒自适应攻击仍可削弱分离性。

# 6. 实用指南
论文及代码已开源：GitHub `jsrdcht/DEFUSE`。复现要点：输入统一为224×224；默认 SDXL、DINOv2、ImageNet-900/无毒公开数据；扩散训练分辨率512、AdamW、学习率 \(10^{-4}\)、batch 64、1 epoch；DDIM约20–30步；默认阈值 \(\tau=0.1\)。迁移到其他编码器时，只需替换 \(f\)、调整特征维度投影，并在目标域少量无标注图像上适配；医学、遥感等 OOD 域仍可迁移，但应重新校准阈值。

# 7. 总结
**核心思想：**用语义重建暴露后门表示。

**速记版 pipeline：**
1. 用可疑编码器提取图像表示；  
2. 将表示转换成生成模型条件；  
3. 用扩散先验生成重建图；  
4. 比较原图和重建图的语义；  
5. 相似度过低则标记为后门。

**Key Findings:**

- To address these limitations, we propose DEFUSE, a generalizable backdoor detection framework for SSL encoders.
- Extensive experiments demonstrate that DEFUSE substantially outperforms existing detectors across diverse attack settings, generalizing to both visual SSL and vision-language encoders.
- Notably, our method greatly reduces the reliance on prior knowledge about the victim encoder or the attack strategy.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.25851v1)
- [arXiv](https://arxiv.org/abs/2608.25851v1)

---

