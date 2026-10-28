time: 20261028

# Arxiv Computer Vision Papers - 2026-10-28

## Executive Summary

## 本期研究趋势

本期 10 篇论文集中在机器人世界模型、视频生成与动作控制的融合，以及如何让仿真更贴近真实执行。多项工作试图学习更可预测、可供规划使用的潜在状态，并以动作条件、跨具身表示或统一 token 接口连接视频预测和机器人控制；TAPESTRY 则探索单一骨干同时支持视频生成与控制。另一条主线是从“生成得像”转向“执行得好”：ExecDiffuser把轨迹候选质量与最终发布选择分开优化，手内位姿到达工作则用独立残差滤波调节任务成功与掉落风险。

场景构建与迁移同样受到关注。SceneAgent 将 3D/4D 采集、物理建模和模拟器内检查修复组织成流水线，再用场景变体生成训练数据；“Pretraining Knows What, Simulation Teaches How”研究预训练知识与仿真技能如何支持组合泛化。总体来看，研究正在从单纯预测或重建，走向预测、规划、控制、仿真数据生成和闭环验证的联合设计。

## 值得优先阅读

- **ExecDiffuser**：清楚展示更好的候选集不必然带来更好的执行，并用反事实选择评估和前缀干预分析闭环影响；证据主要来自仿真。
- **Pushing Dexterity and Safety for In-Hand Object Pose Reaching**：跨四种物体与六种腕向累计 4,800 个真实机器人目标，量化成功率与掉落的折中；同时揭示深度策略在刀和酒杯上的失败边界。
- **SceneAgent**：把多源场景重建、动态 4D 拟合、物理估计和仿真检查串成可修复流水线。重建与动态基准结果显著，但配置运行次数和场景覆盖有限，物理参数尚未实测标定。
- **TAPESTRY 与 RynnWorld-Latent**：适合关注视频—动作统一表示及跨具身世界模型的读者，重点比较其动作接口、条件方式和评估边界。

阅读本期工作时，可特别检查潜在预测是否真正服务于动作选择、仿真结果是否经真实执行验证，以及论文报告的指标能否覆盖长时序和长尾失效。

---

## Table of Contents

1. [DriveUniTok: A Unified Multimodal Tokenizer for Driving World-Action Models](#2610.35044v1)
2. [Pretraining Knows What, Simulation Teaches How: Compositional Generalization in a Sim-Trained Robot Agent](#2610.35817v2)
3. [E0.5: From Reconstructive to Future-Predictive 3D Foundations for Robotic World-Action Modeling](#2610.36011v1)
4. [RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions](#2610.36791v1)
5. [Video World Models Need Predictable Latents](#2610.34663v1)
6. [Adapt for Planning, Not just Prediction: Optimistic Test-Time World Model Adaptation](#2610.35636v1)
7. [ExecDiffuser: Closing the Sample-to-Execution Gap in Diffusion Planning](#2610.36837v1)
8. [TAPESTRY: A Single Backbone for Video Generation and Robot Control](#2610.36823v1)
9. [SceneAgent: An Agentic Pipeline from 3D and 4D Captures to Simulation Environments](#2610.34745v1)
10. [Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well](#2610.37652v1)

---

## Papers

<a id='2610.35044v1'></a>
## [DriveUniTok: A Unified Multimodal Tokenizer for Driving World-Action Models](https://arxiv.org/abs/2610.35044v1)

**Authors:** Jingbo Xu, Hanchao Leng, JiaHui Huang, Hanbing Li, Jingwei Zhao, Kuiyuan Yang, Chen Gao, Guang Chen, Hangjun Ye, Yong Li

**Published:** 2026-10-26

**Categories:** cs.CV, cs.RO

**Abstract:**

World models are emerging as a fundamental component of policy learning for real-world agents, enabling agents to predict future world states and plan actions within a learned representation of the dynamic environment. Existing approaches, however, often model appearance, metric geometry represented by projected LiDAR depth, and ego motion either partially or independently, overlooking the fact that these modalities are different projections of a shared underlying world state. This separation can lead to information loss and cross-modal inconsistency, ultimately limiting downstream performance. Motivated by this unified view of the world, we propose DriveUniTok, a unified multimodal tokenizer for driving world-action models (WAMs) that encodes these modalities in a single decodable continuous latent. We freeze DriveUniTok and adapt a pretrained diffusion model, DriveUniWAM, to predict future unified latents; the tokenizer then decodes them into future appearance, projected LiDAR depth, and ego trajectories. Experiments on NAVSIM demonstrate that our unified design achieves better future prediction and planning performance at lower downstream computational cost than separate-stream alternatives. With reinforcement learning post-training, DriveUniWAM achieves a PDMS of 91.54, reaching state-of-the-art performance among the evaluated WAM-based planners.

### 论文解读

#### 摘要翻译
驾驶世界模型既要预测画面，也要理解空间几何和车辆自身运动。现有方法常把 RGB、激光雷达深度和自车轨迹分开编码，或让预测器在不同表示之间自行学习对应关系，容易造成信息损失和模态不一致。本文提出 DriveUniTok，把外观、投影深度与相对位姿放进同一可解码连续潜表示，并用它驱动未来场景和轨迹联合生成。NAVSIM 实验显示，该表示在预算匹配时改善预测与规划；经奖励后训练的模型 PDMS 达到 91.54。

#### 方法动机分析
驾驶环境中的图像、几何和自车运动是同一动态场景的不同观测。若它们分散在独立潜空间，世界动作模型必须同时学习模态间对齐和随时间演化；若只用视觉潜变量，几何尺度和轨迹状态可能表达不足。作者据此提出核心假设：用一个共同、可解码的潜变量承载三种信号，能让未来预测和规划共享状态表示，并降低单独建模各模态的计算负担。

#### 方法设计详解
输入为每个以 2Hz 采样的五帧片段，输出为统一潜变量及其可解码的 RGB、深度和位姿。编码器（encoder）模块中，RGB 图像和投影到相机平面的稀疏 LiDAR 深度经冻结的 DINOv3-L 提取特征，位姿由可训练编码器变成 8 个 ego tokens。编码器使用 576 个空间查询和 8 个 ego 查询，在自注意力与 RGB、深度可变形交叉注意力中共同聚合信息；时间压缩器将每个查询的多帧状态聚合，经变分瓶颈生成 584 个、宽度为 16 的统一 tokens。共享解码器从同一潜表示重建 RGB、深度、相对位姿，并通过训练期语义分支对齐 DINOv3 特征。随后冻结 tokenizer，以预训练 Wan2.2 扩散 Transformer 为基础，通过 flow matching 预测未来统一潜变量；给定历史和导航条件，解码结果同时包含未来视觉、几何与轨迹。规划奖励后训练在 NAVSIM 上对候选未来打分，仅更新去噪器中的 LoRA 参数。

#### 方法对比分析
区别于 RGB VAE 加独立动作头，或各模态各自编码的方案，DriveUniTok 在编码阶段即让查询跨模态交互，并让每个预测 token 都可被解码为多种目标。作者提供自然预算与等预算两类分离流对照。等预算保证 token 数、潜变量标量数、序列长度和下游计算大致一致，但统一模型与分离模型的 tokenizer 参数量并不相同，因此它同时体现共享参数和交互结构的效果。它适合需要一起预测场景和车辆轨迹的驾驶世界动作模型。

#### 实验分析（精简版）
在 NAVSIM 等预算比较中，统一表示的 PDMS 从 87.5 到 91.0；未来深度 AbsRel 降低到 0.1059（分离流对照为 0.1850），轨迹 ADE 降低到 0.3816（对照为 0.4186）。自然预算分离模型使用 1,160 个 tokens，统一模型用 584 个；每步 DiT 计算从 34.2 TFLOPs 降至 16.3。奖励后训练把 PDMS 提至 91.54。基准为非交互式规划评估，不能据此推断真实道路闭环安全。

#### 实用指南
复现时需保留 RGB、深度和位姿的同步与相对坐标定义，并分别报告 tokenizer 与预测器的参数及计算量。论文 tokenizer 使用 33K 次更新；下游 DriveUniWAM 训练 130K 次更新，AdamW 学习率为 10⁻⁴。应同时比较传感器条件和统一/分离预算；模态缺失测试只在评估时屏蔽输入，不等于训练过的传感器故障鲁棒性。推广到真实车辆或其他驾驶数据仍需新实验。

#### 总结
核心思想：让未来共享同一状态
1. 将图像、深度和位姿共同编码。
2. 在统一潜空间预测未来场景与轨迹。
3. 用规划奖励微调未来生成策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.35044v1)
- [arXiv](https://arxiv.org/abs/2610.35044v1)

---

<a id='2610.35817v2'></a>
## [Pretraining Knows What, Simulation Teaches How: Compositional Generalization in a Sim-Trained Robot Agent](https://arxiv.org/abs/2610.35817v2)

**Authors:** Benjamin Pikus, Daniel Saunders, Mircea Mironenco, Salman Mohammadi, Mathew Mathew, Sirou Zhu, Divya Makkar, Xuhua Huang, Qinbo Li, Supreeth Narasimhaswamy, Siyuan Cen, Dejia Xu, Shyamal Buch, Thomas Neff

**Published:** 2026-10-26

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG

**Abstract:**

A robot policy needs to understand what a task is asking for in a scene, and how to move the robot to accomplish that task. To achieve these two skills, we introduce Zero-real Agent Policy (ZAP), a recipe that teaches a small off-the-shelf vision-language model to control a robot via a simple tool harness. We use a pretrained 27B parameter model to bootstrap world understanding and reasoning, followed by supervised fine-tuning and GRPO on simple simulated tasks to teach it how to move the robot. Without observing real-world robot data during training, ZAP reaches 96.7% success on real-world tasks, matching GPT-6 Astra in our harness and outperforming the MolmoAct2 VLA (33.3%), while running 3x faster and 17x cheaper than GPT-6 Astra. Furthermore, ZAP also generalizes to an unseen embodiment as well as new objects, goals and strategies never seen during training. On a single-arm Franka, ZAP reaches 55.9% on RoboLab-120, 4.9 percentage points ahead of the leading VLA/WAM baseline.

### 论文解读

#### 摘要翻译
机器人操作既需要理解任务、物体和空间关系，也需要把理解转化成准确动作。本文提出 ZAP：先用预训练视觉语言模型提供语义理解，再完全在仿真中通过成功轨迹微调和强化学习教它运动。训练不使用真实机器人数据。作者报告，该策略能迁移到真实双臂任务，并在新物体、新目标、新策略和新机器人条件下组合推理与运动。

#### 方法动机分析
端到端视觉语言动作模型依赖机器人动作数据，可能难以覆盖新的指令组合；直接调用大型智能体虽有较强推理能力，却响应慢、成本高。ZAP 的假设是，语义知识可以从视觉语言预训练继承，运动技能则能通过简单、统一的工具接口在仿真中学习。若仿真和实机提供一致的观察与动作格式，策略可能无需真实示范就迁移。

#### 方法设计详解
模型从 27B 参数视觉语言模型初始化。统一 harness 将相机画面、分割掩膜、物体三维中心和机器人状态整理为输入，提供查询物体、维护短计划、移动末端执行器、设置夹爪和结束任务等工具。动作是绝对末端位姿，不提供抓取宏或固定技能，接近、抓取、抬升、搬运和释放由模型组合。首先进行短程 GRPO 暖启动，从自身成功轨迹中筛选数据做监督微调；难任务通过前缀树搜索复位场景、重放部分成功轨迹并扩展分支。随后在 Isaac Sim 的 40 个简单抓放、堆叠和转向任务上做 GRPO。奖励根据接近、抓住、抬高、靠近目标给予逐级部分分，同时显著提高完成奖励，并惩罚无效动作。仿真和硬件共用接口，机器人差异由具身配置及底层控制器处理。

#### 方法对比分析
与从机器人动作中同时学习语言理解和动作的 VLA 不同，ZAP 冻结语义知识来源于预训练，主要在仿真中学习运动；与依赖人工编写程序或技能库的方案相比，它直接通过完整末端位姿工具组合动作。它仍需 perception、运动控制和执行结果反馈等 harness 组件，不能脱离接口单独理解为端到端模型。适用方向是具有稳定几何工具接口、可模拟训练且能在硬件侧复用控制抽象的操作任务。

#### 实验分析（精简版）
真实双臂评估覆盖 10 个任务、每任务 3 次试验。ZAP 成功率为 96.7%，与同一 harness 中的 GPT-6 Astra 相同；平均任务耗时 146 秒，对照为 397 秒，估算费用为 0.29 美元，对照为 4.87 美元。单臂 Franka 数字来自 RoboLab-120 仿真评测，ZAP 为 55.9%，并非真实 Franka 部署。结果展示迁移潜力，但真实任务和试验次数有限，且对照成绩会随 harness 改变。

#### 实用指南
复现关键在于仿真与实机共用相同观察、坐标单位、动作格式、校验和失败反馈。论文用逐级奖励解决早期零成功导致无梯度的问题，并从模型自身成功轨迹构建微调数据。作者报告 GRPO 使用 40 个任务，每批 640 个 episode；可先检查奖励是否鼓励部分完成而非完整目标，再评估工具无效率。迁移前需分别测量感知误差、控制器成功率和任务成功率，扩大真实试验覆盖。

#### 总结
核心思想：仿真教动作，预训练供理解
1. 统一仿真与实机工具接口。
2. 从仿真成功轨迹微调并强化学习。
3. 在新任务中组合语义理解与运动。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.35817v2)
- [arXiv](https://arxiv.org/abs/2610.35817v2)

---

<a id='2610.36011v1'></a>
## [E0.5: From Reconstructive to Future-Predictive 3D Foundations for Robotic World-Action Modeling](https://arxiv.org/abs/2610.36011v1)

**Authors:** X-Era AI Lab Team,  :, Jianhao Lai, Zhihao Zhan, Zhiquan Liao, Qichang Li, Shubin Chen, Zhiyuan Liu, Hongsen Lei, Jianman Lin, Xiaoxin Lin, Ke Gong, Baiyue Gu, Shiming Chen, Junlin He, Longqi Yang, Junrong Xiao, Haolin Luo, Xiaojuan Su, Jianfeng Li, Yu Luo, Xu Sun, Likui Zhang, Jusheng Zhang, Qinhan Lyu, Tianshui Chen, Liang Lin, Guangrun Wang

**Published:** 2026-10-26

**Categories:** cs.RO

**Abstract:**

World-action models (WAMs) offer a promising framework for embodied intelligence, yet mainstream approaches remain centered on video foundation models or vision-language models. Even when geometric features extracted by 3D models such as VGGT are incorporated, they are typically passed into a video model or VLM that serves as the primary backbone, leaving 3D geometry as an auxiliary input or prediction target rather than the native foundation for world modeling. We present E0.5, a geometry-native world-action model built directly on 3D foundation models. E0.5 introduces two key innovations. First, it adopts a 3D foundation model as the primary backbone, making 3D geometry the native representation for both world modeling and action prediction. Second, it explicitly predicts future 3D states together with actions, enabling the model to capture geometric world dynamics beyond action-only modeling or future video prediction. Experiments on RoboTwin 2.0 first demonstrate the effectiveness of E0.5 without embodied-data pretraining, achieving success rates of 89.90\% and 50.09\% under clean and randomized settings, respectively. Building on these results, we further pretrain E0.5 on more than 160k robot trajectories from AgiBotWorld-Beta, jointly learning robot actions and future geometry within the pretrained 3D foundation. This embodied pretraining improves mean real-world manipulation success rates by approximately 15 and 7 percentage points on AgileX Piper and Zerith H1, respectively. Together, these results demonstrate the potential of geometry-native foundations to acquire robot interaction knowledge and transfer it to physical manipulation. E0.5 embodies this geometry-native approach as a member of the VWA family of world-action models.

### 论文解读

#### 摘要翻译
世界动作模型需要表示当前场景，也要理解动作会如何改变场景。现有方案多以视频或视觉语言模型为主干，即便加入几何特征，几何往往仍是辅助信息。本文提出 E0.5，以三维几何基础模型作为主要、可训练的表征，并让模型一边生成动作，一边预测动作片段结束时的未来深度、三维点图和相机参数。作者在仿真中验证该方法，并进一步用大规模机器人数据预训练后，在两种实体平台上测试操作任务。

#### 方法动机分析
仅恢复当前几何无法直接要求模型理解物体和场景将如何变化；预测未来视频则同时涵盖外观变化，监督信号可能分散。作者假设，让几何基础模型预测与动作时域对齐的未来空间结构，会使其学习更适用于控制的动态表示。几何目标与动作共同更新同一主干，而不是单独增加一个与策略表示弱耦合的辅助生成器。

#### 方法设计详解
输入包括多视角当前 RGB、语言指令和机器人状态，动作为未来 32 步关节及夹爪值。多视角图像由 VGGT-Omega 几何主干编码，语言条件进入跨视角全局计算；几何特征通过逐层注意力桥接到基于 DiT 的动作专家，动作使用 flow matching 生成。训练时，几何头还要预测动作片段终点的深度图、点图和相机参数，几何损失权重为 0.01。未来帧及其几何只作训练监督，推理不需要未来观测。仿真任务在 RoboTwin 2.0 上训练；大规模预训练使用 AgiBotWorld-Beta 的 160,010 条轨迹，报告约 2,645.64 小时数据，之后再对每个实体平台用任务演示单独微调。

#### 方法对比分析
相较于把深度或点云作为视频/VLM 的额外输入，E0.5 将几何基础模型本身设为共享的世界与动作表征。相较只重建当前场景，核心变化是监督动作终点的未来几何；作者的消融显示该目标优于当前几何重建。它适用于几何状态可获得或可估计、且动作结果能以空间结构描述的操作任务。模型仍依赖视觉编码、语言条件与独立动作专家，不能简单理解为单一几何网络直接输出控制。

#### 实验分析（精简版）
RoboTwin 2.0 清洁到随机化评估中，E0.5 不使用具身数据预训练，50 个任务上的成功率为清洁场景 89.90%、随机场景 50.09%。五轮训练的目标消融中，未来几何监督达到 82.72%/29.20%，当前几何监督为 75.94%/20.12%，无几何重建为 73.22%/19.84%（清洁/随机化）。真实机器人每个任务、设置和平台仅 10 次试验；预训练 E0.5 的 OOD 平均成功率在 Piper 为 73.3%，在 Zerith H1 为 76.7%。

#### 实用指南
复现时应区分仿真主结果、较短预算消融和实体机器人评估，避免直接比较不同训练预算。论文将几何监督权重设为 0.01，动作时域为 32 步，推理用 10 步去噪；实体后训练每平台每任务使用 200 条示范、5 个 epoch。大规模数据中的深度由 Depth Anything 3 估计，不是所有视角的传感器真值；迁移时需检查深度质量、相机标定和未来几何误差是否与实际操作收益一致。

#### 总结
核心思想：让三维表征预测动作后果
1. 用三维基础模型编码多视角场景。
2. 联合生成动作并预测未来几何。
3. 用仿真和机器人示范验证操作表现。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36011v1)
- [arXiv](https://arxiv.org/abs/2610.36011v1)

---

<a id='2610.36791v1'></a>
## [RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions](https://arxiv.org/abs/2610.36791v1)

**Authors:** Haoyu Zhao, Yinzhou Tang, Zixuan Wang, Minghao Zhu, Dongchi Huang, Xingyue Zhao, Qin Zhao, Kehan Li, Siteng Huang, Xin Li, Zhenyu Zeng, Mingxiu Chen, Deli Zhao, Zhongyu Li

**Published:** 2026-10-27

**Categories:** cs.RO

**Abstract:**

Learning controllable world dynamics from heterogeneous interaction videos is challenging because robot embodiments use incompatible action spaces, while human videos lack native control labels. We present RynnWorld-Latent, a cross-embodiment video world model that uses the 3D-grounded latent actions introduced by RynnLAM as a shared conditioning interface. This representation enables joint pretraining on human interaction videos, robot demonstrations, and simulation rollouts without requiring a common native action space. Built on a pretrained video transformer, RynnWorld-Latent combines global action tokens with an identity-initialized local pathway that aligns actions with future video latents. We pretrain on 23 datasets comprising approximately 553.5M frames and adapt the model to target robots through post-training. The framework supports two data-generation routes: digital teleoperation uses target robot commands to generate action-video demonstrations, while latent-action synthesis transfers source motion to an edited initial frame and pairs the resulting rollout with retargeted pseudo robot actions. Experiments show improved video prediction metrics on public benchmarks and two real-robot platforms, alongside qualitative action following when different action sequences share the same initial observation. Synthetic mid-training improves policy success rates on LIBERO and LIBERO-Plus, and both generation routes improve real-world policy success rates over real-demonstration-only training in the evaluated settings. These results support universal latent actions as a scalable interface for learning controllable visual dynamics across heterogeneous embodiments.

### 论文解读

#### 摘要翻译
跨机器人学习世界模型时，不同机械臂的关节、坐标和控制方式各不相同；人类交互视频通常也没有机器人动作标注。本文提出 RynnWorld-Latent，以冻结的三维潜在动作表征作为共同条件，让视频世界模型从人类、机器人和仿真数据中学习动作相关的视觉变化。模型还能生成两类训练数据：通过数字遥操作生成动作视频，或把人类视频的首帧编辑成机器人场景后迁移原有潜在动作。

#### 方法动机分析
语言指令能描述“拿起杯子”，却没有规定精细的接触时序和运动轨迹；直接用机器人原始动作训练，又难以跨不同具身形态复用。作者提出用视频潜在动作描述观察之间的变化，试图把低层控制差异抽象掉。核心假设是：若潜在动作与未来视频帧逐段对齐，同一模型就能利用无机器人控制标注的人类视频。该假设面临的重要局限是，潜在变化也可能包含相机运动或外观因素。

#### 方法设计详解
模型基于 Cosmos3 视频 Transformer 和 rectified flow。冻结的 RynnLAM 从相隔四帧的图像中提取连续潜在动作；81 帧片段压缩为 21 个视频 latent，因此动作与未来 latent 可逐帧对应。每个动作先映射成 token 参与全局注意力，同时以零初始化的局部模块注入对应未来帧的视觉 tokens。预训练只对未来视频 latent 去噪，初始帧和动作条件保持固定。适配到目标机器人后，另加具身专属编码器，把该机器人命令送入相同条件路径。数字遥操作以目标机器人动作生成视频反馈；另一条路线则编辑人类视频首帧并迁移原 latent actions，再用重定向和逆运动学生成伪机器人动作。

#### 方法对比分析
相比用任务级文本控制视频生成，这种条件能表达更细粒度的动作变化；相比跨机器人直接共享原始关节动作，它不要求控制维度相同。模型结合全局动作 token 和时间对齐的局部注入，既提供整体运动条件，也将每段动作连接到具体未来帧。两种合成路线分别利用操作者命令和人类视频动作，但生成图像与重定向动作并非来自同一个物理过程，因此配对数据仍需验证。

#### 实验分析（精简版）
预训练集由 23 个数据集构成，约 553.5M 帧。视频预测指标在公开 GR-1 测试集和两个实体平台上有所改善；公开基线取自既有文献，自建平台的测试则是目标机器人后训练后的留出片段，不是未见具身的零样本迁移。80M 合成视频中训练再用真实数据微调后，π0.5 在四个实体任务上的成功率为 94.29%、97.14%、60.00%、68.57%；仅真实训练分别为 82.86%、77.14%、40.00%、45.71%。每项结果为 35 次试验。作者将这些差异视为观察到的提升，不声称小样本下统计显著。仅用数字遥操作数据的表现仍低于真实数据训练。

#### 实用指南
复现时应记录训练集来源、目标平台后训练和合成数据筛选规则；论文视频模型训练为 20K 步、全局 batch 512，视频片段最长 81 帧、对应 20 段潜在动作。推理时固定首帧与输入动作，只对未来视频 latent 去噪，并采用 1.15 的动作条件 guidance。迁移时要把视频预测质量、动作条件响应、伪动作可行性和真实闭环成功分开评估。潜在动作可能混合操控变化、相机运动与外观变化；像素相似度不等同物理正确。动作重定向与生成视频独立完成，时间对齐和筛选也不能保证接触关系一致。合成中训练增加了数据量和训练计算，需采用等算力对照来拆分收益。跨平台应用需要目标机器人后训练，并检查新平台动作的同步、限位和逆运动学误差。

#### 总结
核心思想：用共享潜在动作连接跨具身视频
1. 从视频转场提取连续动作表征。
2. 用动作条件预测未来视频帧。
3. 迁移或遥操作生成合成训练数据。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36791v1)
- [arXiv](https://arxiv.org/abs/2610.36791v1)

---

<a id='2610.34663v1'></a>
## [Video World Models Need Predictable Latents](https://arxiv.org/abs/2610.34663v1)

**Authors:** Mattia Scardecchia, Sherry Yang

**Published:** 2026-10-26

**Categories:** cs.LG, cs.AI, cs.CV, cs.RO

**Abstract:**

Video world models trained on large-scale robot data promise a safe and scalable approach for learning and evaluating policies that act in the physical world. However, they still struggle with long-horizon rollouts and out-of-distribution generalization, and they remain computationally demanding at inference time. Typically, video world models make predictions in the latent space of a reconstruction-trained autoencoder, which however is not directly optimized for dynamics modeling. On the other hand, world models designed for latent planning usually optimize latents to be predictable, but they forgo accurate prediction of observations in pixel space. In this paper, we show that it is possible to achieve the best of both worlds: a latent geometry optimized to make future prediction as simple as possible without sacrificing accurate prediction of future observations. More precisely, we find that temporal objectives for representation learning, paired with a past-conditioned decoder architecture, lead to consistent improvements in generative video world models at the same training budget. The gains include up to 5x cheaper inference, over 7x faster training, and 15-24% lower pixel and perceptual errors out of distribution and over long horizons. Our world models also agree more closely with real-world policy evaluation when using a vision-language model to judge task success from their pixel-space predictions. Overall, our study indicates that temporal predictability is an important design criterion for the latent space of video world models, and provides concrete guidance on how to incorporate it into the training of autoencoders.

### 论文解读

#### 摘要翻译
视频世界模型依赖潜变量预测未来，但重建训练得到的潜变量可能难以预测；仅保留可预测信息又可能丢掉画面细节。本文研究通过时间目标改造潜表示，并让解码器利用过去帧，能否兼顾观测信息、时间可预测性与世界模型的生成质量。

#### 方法动机分析
普通自编码器倾向保存像素中变化丰富的细节，其中可能包含与动作状态无关的随机信息，增加世界模型预测负担。相反，单纯追求未来预测会偏向稳定、易预测的部分，牺牲重建。作者的核心假设是：以时间目标塑造潜变量，同时把过去帧交给解码器补足静态外观，可以缓解两者冲突。

#### 方法设计详解
输入为相邻视频帧及动作。第一步，用重建损失和时间损失的加权和微调图像自编码器；时间损失分别尝试未来潜变量预测、像素预测、对比式预测或逆动力学（从相邻潜变量预测动作）。第二步，解码下一帧时把前一帧作为上下文，使解码器可复用静态背景，预测潜变量集中描述变化。第三步，冻结编码器，在潜空间训练动作条件扩散世界模型。推理时给定起始帧与动作序列，模型自回归生成潜变量，再由带上下文的解码器还原视频。真实视频实验在 256×256、5Hz 的 Bridge V2 片段上进行；等预算设置包含 80,000 次自编码器更新和 100,000 次世界模型更新。

#### 方法对比分析
与只用重建的常见流程相比，本文在表示学习阶段加入时间监督，再通过过去条件解码器把稳定信息从潜变量中分担出去。未来潜变量预测在部分长时与少步预测指标上突出；逆动力学在若干图像和分布指标上更优，因此没有单一目标包揽所有指标。单独使用过去条件解码器也有帮助，但时间目标带来的增益更明显地出现在困难预测条件。方法适合关心动作条件视频预测、较长滚动或有限推理步数的研究。

#### 实验分析（精简版）
作者在合成环境和机器人操作视频上评估，比较重建质量相同与训练预算相同两种设定。在三种子等重建质量实验中，潜变量预测将分布外像素 MSE 从 33.87 到 29.14（数值按论文定义乘以 10⁻³），重建基线为比较对象；但需要 156.5K 次自编码器更新，而基线为 4.4K。策略评估显示，少步采样时加入时间目标可更好保留与真实策略表现的相关性，不过该评估只用一个训练种子和视觉语言模型裁判，适用范围有限。

#### 实用指南
复现时应同时报告自编码器和世界模型训练成本，并在等重建质量、等预算两种条件下比较。需分别检查分布外、长滚动和少采样步数，且调节时间损失权重：较强权重虽能提升部分长时预测，却可能损害图像分布质量。作者说明实现代码可获取；迁移到其他视频领域或三维时空压缩仍需实证，机器人/自动驾驶中的潜在用途属于待验证方向。

#### 总结
核心思想：让潜变量更懂时间
1. 用重建与时间目标共同训练编码器。
2. 用过去帧帮助解码器恢复稳定外观。
3. 在冻结潜空间训练动作条件世界模型。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.34663v1)
- [arXiv](https://arxiv.org/abs/2610.34663v1)

---

<a id='2610.35636v1'></a>
## [Adapt for Planning, Not just Prediction: Optimistic Test-Time World Model Adaptation](https://arxiv.org/abs/2610.35636v1)

**Authors:** Akshay Mete, Dileep Kalathil, P. R. Kumar

**Published:** 2026-10-26

**Categories:** cs.LG, cs.AI, cs.RO

**Abstract:**

Latent world models are typically pre-trained offline and kept frozen at test time, so planning with them can degrade under distribution shift. Test-time adaptation (TTA) can mitigate this degradation by updating the model during deployment. Unlike pre-training on fixed data, TTA for planning is closed-loop: the planner uses model rollouts to select actions, and the resulting transitions drive the next update. This raises the question: which objective should guide TTA for planning? Under a prediction-only objective, the model is fit only to the outcomes of its own actions, so it can remain wrong about the actions that would reach the goal, and low prediction error need not translate into better planning. Inspired by reward-biased maximum likelihood from adaptive control, we introduce Optimistic Test-Time Adaptation (OpTTA). Its objective has two terms, the prediction loss on observed transitions and the planner's goal-reaching cost on its candidate rollouts, both minimized over the model parameters. The planning term biases adaptation toward models that are optimistic about reaching the goal, without any additional environment interaction. Across visual and environment shifts on manipulation, reaching and navigation tasks, OpTTA raises final planning success over the frozen model by 13.5 percentage points on average, compared with 6.9 for the prediction-only objective (AdaJEPA), when only the predictor is adapted. When the encoder is adapted as well, the gains are 13.5 and 7.5 points. These experiments span three JEPA world models with different encoders and training objectives and, for temporal straightening, both CEM and gradient-based planners. Our results suggest that test-time adaptation of world models for planning benefits from an optimistic, task-aware term in addition to the prediction loss.

### 论文解读

#### 摘要翻译
世界模型常在部署时保持冻结，遇到视觉或环境变化后，规划效果可能下降。现有测试时适应方法根据真实执行产生的转移来降低预测误差，但规划器本身又依赖模型想象来选动作，因此模型可能只学会解释自己已选动作，仍误判哪些动作能接近目标。本文提出 OpTTA，把观测转移的预测损失与候选想象轨迹的目标代价一同用于适应，鼓励模型对达成目标的路径保持乐观。

#### 方法动机分析
训练目标与规划目标不一致是关键痛点。预测准确并不保证规划成功，尤其在闭环部署中，数据由当前模型和规划器共同产生。作者借鉴 reward-biased maximum likelihood 的思想，假设在预测误差约束下直接优化世界模型的目标可达代价，能减少这种目标错配；它不要求额外学习奖励函数或校准不确定性。

#### 方法设计详解
每个任务开始时恢复预训练参数并清空转移缓存。循环中，规划器根据当前潜状态和目标图像生成候选动作序列，执行最佳序列的第一步，把新观测转移加入缓存。接着最小化 OpTTA 损失：一项是在近期真实转移上的潜变量一步预测误差；另一项是在规划器已有候选序列上的目标距离代价。候选动作保持固定，规划项梯度穿过模型预测轨迹更新世界模型参数，而非调整动作本身。两项损失的梯度范数用于自适应平衡权重，乐观程度随重规划次数逐渐增加。适应后重新规划。实验中 CEM 每次规划 5 个潜步、只执行第一个潜步、缓存最近 5 个转移，每次重规划做 1 次 Adam 更新，最多 30 次；模型可只更新预测器，也可同时更新编码器。

#### 方法对比分析
AdaJEPA 只使用观测转移预测误差，OpTTA 在相同初始化、规划器、缓存、更新参数和更新频率下加入规划目标，因此对照聚焦在适应目标。与显式不确定性方法不同，OpTTA 通过目标函数引入乐观倾向。它适用于能够对想象状态计算目标代价、且模型梯度可用的规划问题；候选轨迹来自 CEM 最后一次迭代，规划项不会反向改变动作序列。

#### 实验分析（精简版）
实验覆盖三种 JEPA 世界模型、操作/到达/导航任务、视觉扰动和环境变化，并使用 CEM 与梯度规划。在 15 个分布变化设置中，OpTTA 有 14 个设置优于预测式适应。仅适应预测器时，其相对冻结模型的平均最终成功率提高 13.5 个百分点，AdaJEPA 为 6.9 个百分点。TwoRoom 视觉扰动是失败例：两种适应方法均可能低于冻结模型。OpTTA 还增加每次重规划的计算开销。

#### 实用指南
复现时应按任务重置模型，并匹配对照方法的缓存大小、更新参数和规划预算。论文主要设定为最多 30 次重规划、CEM 200 个候选、30 个精英、10 轮迭代；乐观权重基值为 1，不针对任务单独调参。应同时报告最终成功率、各轮平均成功率和单次重规划耗时。若候选集很大，可测试只用精英轨迹的计算取舍；实际迁移需特别检查目标代价是否会被模型利用。

#### 总结
核心思想：让适应目标对齐规划
1. 用执行转移校正预测模型。
2. 对想象轨迹加入目标代价。
3. 更新模型后重新规划并执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.35636v1)
- [arXiv](https://arxiv.org/abs/2610.35636v1)

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
扩散驾驶规划器通常按生成的候选轨迹质量评价，但车辆每轮只能执行一个轨迹，执行结果还会影响下一轮规划。本文把候选质量与实际选中轨迹之间的差异定义为“候选到执行落差”，提出 ExecDiffuser：IGD 通过执行条件监督改进候选，EGS 从当前可观测信息中选择要执行的候选，并把已发布轨迹作为下一轮上下文。作者在 nuPlan、InterPlan 和 NAVSIM 仿真中评估整体效果，也用反事实执行和前缀干预研究选择质量如何影响后续状态。

#### 方法动机分析
扩散模型即使生成了很好的轨迹，也不能保证选择器会把它交给控制器；继续提升候选集可能反而扩大选择错误。这个落差来自两个阶段依据不同：候选训练常依赖与专家轨迹的模仿相似度，闭环发布则需要考虑道路可行性、路线、风险、进度、舒适度和历史执行。作者的关键假设是，如果候选责任分配、轨迹修正和发布选择共享执行证据，规划器就能更可靠地把“可用的好候选”转化为“实际执行的好决策”。

#### 方法设计详解
每个规划周期使用当前场景、地图、路线和上一发布轨迹生成候选，并计算在线证据。硬筛选只检查驶出可行区域或即将碰撞；其他风险、进度和舒适度作为软偏好。IGD 用执行条件训练偏好给不同 intent query 分配责任，再用可微几何损失修正轨迹，并学习 query 使用和候选数量。EGS 则编码候选几何、当前场景、路线与风险证据、连续性上下文，预测候选效用并在有效集合中发布最高分轨迹。被选轨迹的未执行后缀会对齐到下一周期自车坐标，作为新规划上下文。候选全部无效时系统调用 jerk 限制的 MRM。

#### 方法对比分析
常见生成规划器（如 Diffusion Planner、FlowDrive）主要靠改善模仿学习候选或采样多样性来提升最终轨迹；这里的消融显示这并不保证执行收益：把 IGD 接到固定 imitation 选择器上，最佳候选分数虽提升 1.03，发布分数反降 0.18，regret 从 0.51 增至 1.72。相反，在固定 proposal 的条件下只替换选择器，EGS 将 oracle top-1 recall 从 56.4% 提至 86.6%，regret 从 2.50 降至 0.62。这一分解支持作者的核心设计判断：候选改进和发布选择必须协同训练，单独报告 oracle 候选质量会掩盖执行端失败。IGD 用执行相关责任分配与可微整形训练候选；EGS 学习效用排序，使用当前场景、路线、风险与历史发布上下文。oracle 仅用于离线诊断，绝不输入 EGS。前缀干预执行不同长度的已选前缀后再调用同一冻结规划器，因而能检验反馈链路效应，但仍是仿真中的受控干预，不等同于道路实测。

#### 实验分析（精简版）
五种子统一复现下，ExecDiffuser 在 nuPlan 平均 85.62，比 Diffusion Planner 高 4.02、比 FlowDrive 高 1.31；InterPlan 分数为 38.35，对照为 24.71 和 36.96。固定 imitation 选择器时，IGD 提高最佳候选分数 1.03，却使发布分数下降 0.18，regret 从 0.51 升至 1.72。固定 proposal 时，EGS 将 oracle top-1 recall 从 56.4% 提升至 86.6%，regret 从 2.50 降至 0.62。执行不同初始前缀后恢复同一规划器，EGS 的后续分数优势在 0.5 至 4 秒前缀间由 0.51 增至 1.90，配对区间均不含零。NAVSIM 五种子平均 PDMS 为 92.6，但与 91.2 的 DiffusionDriveV2 是论文结果对比，非统一复现。

#### 实用指南
训练使用 nuPlan v1.1 训练集，另留场景互斥的 10% 校准集；扩散训练/推理步数为 1000/25，AdamW 学习率 5×10⁻⁴，有效 batch size 1024、训练 500 epochs，使用 8 张 RTX 4090D。nuPlan 主表对五个独立种子取算术平均；NAVSIM 则从头训练五个模型，不用外部数据或预训练权重。部署推理在单张 RTX 4090D、batch 1 测得 59.5 ms，较基线增加约 5.2M 参数。复现时要把候选 oracle、硬有效率、publication regret、实际闭环分数和全无效回退分别报告，并按场景而非相关的连续规划帧做配对统计。风险预测用于筛选，但遮挡路口 false-valid 为 5.14%，且 forecast 未覆盖全部必需 MRM 时刻。MRM 是针对检测到的失效的回退，不是形式化安全保证。现有证据来自 nuPlan、InterPlan 和 NAVSIM 模拟，应在真实车载时延和长尾交通场景中继续检验。

#### 总结
核心思想：优化最终发布的轨迹，而非只优化候选
1. 用执行相关证据训练候选和选择。
2. 选择轨迹并回传为下一轮上下文。
3. 用反事实和前缀干预验证闭环效果。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36837v1)
- [arXiv](https://arxiv.org/abs/2610.36837v1)

---

<a id='2610.36823v1'></a>
## [TAPESTRY: A Single Backbone for Video Generation and Robot Control](https://arxiv.org/abs/2610.36823v1)

**Authors:** Jeremy A. Collins, Seungjae Lee, Rachanon Wachakorn, Krishnan Srinivasan, Animesh Garg, Vitor Campagnolo Guizilini, Paarth Shah

**Published:** 2026-10-27

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG

**Abstract:**

Pretrained video generation models encode rich priors about physical interactions, but leveraging them for robot control typically requires either resource-intensive video generation at test time or action heads trained from scratch that do not fully utilize the pretrained backbone. We introduce Tokenizing Actions as Video Patches in a Shared Transformer Policy (TAPESTRY), a world action model that repurposes a pretrained video diffusion transformer into a robotic policy without requiring video generation at inference. Our key idea is to treat robot actions as single-patch tokens in the video model's latent space. A lightweight linear projection maps each action timestep to the patch dimensions of the pretrained VAE, and the resulting tokens are processed by the same patchifying layers and transformer backbone used for video. The shared backbone is optimized through three diffusion objectives sampled per minibatch element: Video diffusion uses VAE latents from robot and human videos to establish dynamics priors. Joint video+action diffusion trains on robot demonstrations using a coupled diffusion timestep and an asymmetric attention mask. Action-only diffusion denoises actions conditioned on clean image latents. At test time, the model generates a compact action token sequence in a single DDIM step from pure noise, producing smooth action chunks at 84 ms per action chunk on consumer hardware and avoiding the cost of full video generation. Across simulation benchmarks and real-robot experiments, TAPESTRY achieves strong performance with substantially lower inference cost than comparable joint video--action policies. More information can be found at https://tapestry-wam.github.io.

### 论文解读

#### 摘要翻译
视频生成模型包含丰富的交互先验，但将其用于机器人控制，常需在推理时生成视频，或另训一个无法充分利用预训练权重的动作头。本文提出 TAPESTRY，把每个动作时刻编码成类似视频 patch 的 token，送入同一个预训练视频扩散 Transformer。通过视频扩散、视频与动作联合扩散、动作独立扩散三种目标共同训练，部署时只运行短动作序列，并用单步 DDIM 生成动作块。

#### 方法动机分析
作者认为，机器人策略需要的是视频主干中的动态知识，不一定要在执行时生成整段像素视频。传统视频到动作方法要么先生成视频、再用逆动力学取动作，推理成本和误差会叠加；要么只抽取主干特征，再训练独立动作头，动作没有直接通过视频主干的 patch 表征。TAPESTRY 的假设是：让动作 token 的统计和输入路径都接近视频 latent patch，就能更直接复用预训练 DiT 权重，同时减少推理序列长度。

#### 方法设计详解
处理流程的输入是当前观测、语言指令、动作片段和训练视频；冻结的视频 VAE 把片段压成 latent，动作 tokenizer 模块用一个线性投影将每步动作映射为单个 patch，再经无仿射 LayerNorm 匹配视频 latent 的均值方差，最后复用视频 patch 卷积。反向读出动作时复用 patch unembedding 和投影矩阵的转置，动作通路新增参数为 1280。共享 CogVideoX-2B 主干按 minibatch 样本混合三类扩散训练：视频目标使用人类和机器人视频；联合目标用同一噪声时刻训练视频与动作，并通过非对称注意力让动作关注未来视频 token；动作独立目标则在当前图像与文本条件下只去噪动作。最终 decoder 读出动作：部署不生成视频，16步动作序列由一个 DDIM 前向直接预测。

#### 方法对比分析
相比生成未来视频后再估计动作，TAPESTRY 不需视频解码器参与执行；相比在视频特征上另接动作头，它的动作 token 直接穿过预训练 patchifier 与 Transformer。与一次预测多个未来视频 token 的联合策略相比，推理仅处理短动作块，因而成本低。这个效率来自特定设计组合：单个动作 patch、动作专属训练目标，以及零终端信噪比下的 v-prediction 单步更新；换扩散参数化后不能直接假设同样成立。

#### 实验分析（精简版）
LIBERO 实验只在 LIBERO-90 提供动作标签，其他四个测试套件仅提供视频，属于视频到动作迁移而非标准全监督测试。TAPESTRY 平均成功率 76%，Cosmos Policy 和 DreamZero 各 70%，从头训练版本为 58%。真实 YAM 机器人三个训练任务各跑 10 次，TAPESTRY 成功 28/30，行为克隆为 15/30。三个新物体位置、动作原语或物体组合条件各 5 次，TAPESTRY 完整成功 10/15，DreamZero 为 4/15；作者指出该差异未达统计显著。RTX 5090 上 16 步动作块耗时 84 毫秒、显存 9GB；DreamZero 同卡优化实现约 3.7 秒，但优化包含发布版没有的缓存与激活检查点。

#### 实用指南
复现时应保持视频、动作数据、训练算力和推理口径可比，并报告去掉视频目标、动作独立目标和共享 tokenizer 的消融。模型单步输出近似动作条件分布的均值，可能不适合具有多个等价动作方案的任务。其单步性质依赖零终端信噪比与 v-prediction；作者的 Wan rectified-flow 对照不具备同样的噪声无关终点。真实 OOD 每种情形仅 5 次试验，结论应视为初步证据。

#### 总结
核心思想：让机器人动作复用视频 patch 主干
1. 将每个动作时刻编码成一个 patch token。
2. 用三类扩散目标联合训练共享主干。
3. 单步去噪动作块，不生成未来视频。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.36823v1)
- [arXiv](https://arxiv.org/abs/2610.36823v1)

---

<a id='2610.34745v1'></a>
## [SceneAgent: An Agentic Pipeline from 3D and 4D Captures to Simulation Environments](https://arxiv.org/abs/2610.34745v1)

**Authors:** Luke Hollis, Tianxing Fan, Heng Yang

**Published:** 2026-10-26

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Visual captures provide detailed scene appearance, but converting them into interactive simulation environments requires object reconstruction, physical modeling, and correction of intermediate errors. We present SceneAgent, an agentic pipeline that coordinates these steps through simulator-based visual inspection and repair. The pipeline processes 3D Gaussian Splatting (3DGS), photogrammetry, LiDAR captures, AI-generated 3DGS scenes, and 4D videos of dynamic scenes, and combines object segmentation, physical parameter estimation, part decomposition, articulation, and deformable simulation. To support policy learning, SceneAgent generates diverse object variants ("digital sisters") and scripted demonstrations. On a cluttered bin scene from GraspClutter6D, SceneAgent raises F1 at 1 cm for every coding agent with scorable outputs both with and without its skills, and its reconstructions are better than SimFoundry: all agent configurations exceed SimFoundry's F1 score, with a best of 60.75% versus 6.9%. On three dynamic scenes from 4DCodeBench, SceneAgent raises the mean dynamic IoU of Gemini 4 Argon from 0.86% to 99.09%. We further demonstrate diverse simulated interactions and preliminary policy transfer to a real robot.

### 论文解读

#### 摘要翻译
SceneAgent 把 3D 或 4D 视觉采集转为可交互仿真环境，协调场景重建、对象分割、物理属性估计、部件拆分、关节/软体建模与导出，并通过模拟器渲染检查和修复中间结果。它支持 3D 高斯泼溅、摄影测量、LiDAR、生成场景和动态视频，还能生成对象/场景变体与模拟示范。料箱场景的重建 F1、三个合成动态场景的 dynamic IoU 均有提升；作者也展示了初步的实机策略微调。

#### 方法动机分析
仅有逼真的视觉重建无法直接支持接触操作：网格尺度、对象分割、物理属性、关节和坐标变换任一环节出错，最终环境都可能不可用。单独调用多个模型并不能自动定位这些错误，因此核心挑战是跨步骤的协调和检查。SceneAgent 让代理在目标模拟器中查看渲染结果并修正转换产物，再把接受/拒绝记录和场景文件传给后续步骤，试图把偶发的人工重建过程变成可复用流水线。

#### 方法设计详解
静态图像先经 COLMAP 和 Splatfacto 重建相机与 3DGS；SAM 掩码与高斯投影规则确定对象点，SAM 3D 生成网格并经 trimmed ICP 对齐。OpenCLIP、DINOv2、每高斯 MLP 与 VLM 投票估计物理类别和属性，之后将 USD 场景导入模拟器检查。PartSAM/CubePart 处理部件，Articulate Anything 提议关节，数字姐妹扩展对象/场景。4D 视频则估计相机、动态材料初态与模拟参数，以渲染序列对比早中晚帧并迭代修正。策略训练流程由脚本专家产生模拟示范，再用于预训练 π0.5 的 LoRA 微调：Gemma-2B 用 rank 16、动作专家用 rank 32，训练 20k 步、batch size 32，单张 H200 约 6.5 小时。policy autoresearch 可把真实失败转成模拟场景变体。

#### 方法对比分析
相较 SimFoundry 等专注静态场景重建的系统，SceneAgent 强调代理检查/修复、光栅化与物理建模的统一流程，并扩展到 LiDAR、生成式输入、4D 动态、变形/液体/断裂模拟、混合场景和策略闭环。GraspClutter6D 料箱场景上，六种 harness 使用 skills 后 F1 为 40.45%–60.75%，不使用时为 2.36%–9.51%，SimFoundry 为 6.9%；但每种配置只转换一次。物理属性由 VLM 标签训练的预测先验提供，而非实测辨识；4D 参数拟合也不唯一。因此系统更像有视觉质检的工程流水线，不是已验证的物理数字孪生。

#### 实验分析（精简版）
GraspClutter6D 对 20 个被遮挡物体计分，使用 SceneAgent 后所有 harness 的 1 cm F1 均显著提升。4DCodeBench 三个合成动态场景中，Gemini 4 Argon 平均 dynamic IoU 从 0.86% 升到 99.09%，其余四组也改善。真实策略迁移仅测试一个实验室 pick-and-place 任务、50 个布局：特定多阶段数据策略成功 44/50（88%），基线为 11/50（22%）。单纯增加示范数不保证更好，40 条示范策略实机成功率 48%，高于 100、200、400 条设置。作者指出静态场景每配置一次、4D 仅三个合成案例，仍需更广评估。

#### 实用指南
复现需串接相机/3DGS、分割、对象网格、物理先验、对齐与仿真器导出，并在组装后对注册和完整场景渲染做检查。USD 可跨 Isaac Lab、MuJoCo 和 Unreal 传递刚体属性；软体、流体和断裂需要各引擎单独配置。把估计的质量、摩擦、刚度当成域随机化起点，不应直接当作标定值。实机策略结果来自模拟示范的 π0.5 LoRA 微调，样本量有限；部署前应对目标对象和场景做独立真实验证。

#### 总结
SceneAgent 把场景重建、物理建模和仿真检查组织成可修复的代理流水线，探索从视觉采集扩展机器人训练数据。
1. 从多种 3D/4D 输入构造可交互场景。
2. 用模拟器渲染检查和修复转换结果。
3. 以对象/场景姐妹和脚本示范支持策略训练。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.34745v1)
- [arXiv](https://arxiv.org/abs/2610.34745v1)

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
作者训练 LEAP 多指手在仿真中把物体连续移动到不同的手内位姿目标，不使用人工示教，并让策略零样本迁移到真实机器人。系统用生成的抓取数据、自适应摩擦课程、舵机辨识和状态到深度的策略蒸馏提升操纵能力；另训练残差安全滤波器，以一个权重调节干预程度。四种物体、六种手腕方向的实机实验累计 13.3 小时、4,800 个目标。状态策略粗容差成功率 95.2%，滤波使掉落次数减少 75%，同时成功率降至 91.5%。

#### 方法动机分析
已有手内操纵多限于旋转或少数手腕姿态，且常用奖励惩罚隐式处理安全。位姿到达还要求精确平移到指尖可达的困难位置，长序列中一次小滑动即可导致掉落。核心挑战是任务成功与安全保持之间的折中。作者把任务策略与安全约束拆开：让任务策略专注达到目标，再让残差滤波学习尽量少改动作但延长无掉落时间。这样的模块化设计让操作者能按场景调保守度，也避免为了安全重新训练完整任务策略。

#### 方法设计详解
每个对象先由 LightningGrasp 生成 10 万初始抓取，再通过 DexterityGen 风格 RRT 扩增至约 42–45 万姿态。策略分四阶段训练：特权状态 PPO 教师、DAgger 状态蒸馏、PPO 学生微调、DAgger 深度图蒸馏。自适应课程先以高指尖摩擦降低探索难度，后逐渐降到真实摩擦，同时让目标从近处扩展到全数据集。BAM 舵机辨识、延迟校准、接触/惯量随机化和深度图增强缩小 sim-to-real 差距。ResSafe 残差策略叠加在冻结任务策略上，训练时输入权重 w，部署时调节它以控制偏离任务动作的程度。

#### 方法对比分析
相较于仅在任务奖励里增加掉落惩罚，独立残差滤波器训练快（约 4 小时）且可调，不必反复训练任务策略。与学习型 HJ 滤波相比，它只需前向计算，不需在线搜索动作；在橡胶鸭模拟比较中，ResSafe 的粗成功率 91.3%、掉落率 4.5%，而相近掉落率的 HCSF 粗成功率 82.0%，LRSF 为 79.3%。代价是它依赖学习数据，并无安全证明；过强滤波也会压制任务动作。实机主实验中状态策略成功率从 95.2%降到 91.5%，而酒杯深度策略加滤波后仍有 40 次掉落，显示收益并不均匀。

#### 实验分析（精简版）
4,800 个目标来自四对象、六腕向、状态/深度、开/关滤波的 96 条序列，每条 50 个目标。状态策略粗容差成功率 95.2%、精容差 75.6%；加滤波后粗成功率 91.5%，掉落由 16/1,200 降到 4/1,200。深度策略粗成功率 71.8%降到 66.7%，掉落由 88 降到 46；刀与酒杯受深度估计长尾误差影响更明显。策略还在未训练手臂运动及物理扰动中保持物体，并完成腕部相机输入的脚本式多步任务；这些任务样本较小，滤波在部分项目表现更差。

#### 实用指南
训练每个对象需要独立策略和滤波器；A1 教师约 15–17 小时，A2/A3/A4 分别约 0.6/2.5–2.8/10 小时，滤波训练约 4 小时，使用 RTX 5090。实机在 RTX 4080 SUPER 上推理约 5.0 ms（状态）或 5.7 ms（深度），加滤波为 5.6/7.4 ms。部署应按对象、传感器、腕部姿态和安全偏好校准 w，分别报告粗/精成功、掉落、掉落前目标数与滤波修正幅度。实机“成功”只要求 10 秒内曾进入容差，不代表保持或安全证明。

#### 总结
一套系统训练配方让多指手在真实硬件上连续完成高难度手内位姿目标，并用残差滤波单独控制掉落风险。
1. 用抓取数据与自适应摩擦课程覆盖可达目标。
2. 用特权教师到状态、深度策略的蒸馏迁移到实机。
3. 将任务能力与可调安全滤波分开评估。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.37652v1)
- [arXiv](https://arxiv.org/abs/2610.37652v1)

---

